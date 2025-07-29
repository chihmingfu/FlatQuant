import transformers
import json
from pathlib import Path
from datetime import datetime

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils
import gptq_utils

def apply_mixed_precision_config(args, model, logger):
    """Apply Report 013 mixed precision configuration"""
    # Report 013: W2 layers: 0, 2, 3, 4, 5, 6, 7, 8; W4 layers: 1, 9, 10, 11, 12, 13, 14, 15
    w2_layers = [0, 2, 3, 4, 5, 6, 7, 8]
    w4_layers = [1, 9, 10, 11, 12, 13, 14, 15]
    
    logger.info("Applying Report 013 Mixed Precision Configuration:")
    logger.info(f"W2 layers: {w2_layers}")
    logger.info(f"W4 layers: {w4_layers}")
    
    # Configure quantizers for each layer
    for layer_idx, layer in enumerate(model.model.layers):
        target_bits = 2 if layer_idx in w2_layers else 4
        
        # Configure attention weights
        if hasattr(layer, 'self_attn'):
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, name):
                    module = getattr(layer.self_attn, name)
                    if hasattr(module, 'weight_quantizer'):
                        module.weight_quantizer.configure(
                            target_bits, perchannel=True, sym=not(args.w_asym), mse=False
                        )
        
        # Configure MLP weights  
        if hasattr(layer, 'mlp'):
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(layer.mlp, name):
                    module = getattr(layer.mlp, name)
                    if hasattr(module, 'weight_quantizer'):
                        module.weight_quantizer.configure(
                            target_bits, perchannel=True, sym=not(args.w_asym), mse=False
                        )
        
        logger.info(f"Layer {layer_idx}: W{target_bits}")
    
    return {'w2_layers': w2_layers, 'w4_layers': w4_layers}

def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)
    
    # Check for mixed precision mode
    mixed_precision_mode = getattr(args, 'mixed_precision', False)
    if mixed_precision_mode:
        logger.info("=== MIXED PRECISION TRAINING MODE ===")
        logger.info("Report 013 Configuration: W2/W4 Mixed + A8KV8")

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)

    # get calibration data
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    logger.info("Finished loading training data.")

    mixed_precision_config = None
    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        
        # Apply mixed precision configuration if enabled
        if mixed_precision_mode:
            mixed_precision_config = apply_mixed_precision_config(args, model, logger)
        
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        flat_utils.reparameterize_model(model)
        logger.info("Finished reparameterize model.")
        
        # Save mixed precision configuration
        if mixed_precision_mode and mixed_precision_config:
            config_data = {
                'mixed_precision_config': mixed_precision_config,
                'model': args.model,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'args': vars(args)
            }
            config_path = Path(args.exp_dir) / "mixed_precision_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Mixed precision config saved: {config_path}")

    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    # Evaluating PPL
    for eval_dataset in ["wikitext2", "c4"]:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
                args,
                eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                hf_token=args.hf_token,
                eval_mode=True
            )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)


    if args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        task_manager = lm_eval.tasks.TaskManager(include_path="./datasets/lm_eval_configs/tasks", include_defaults=False)
        task_names = lm_eval_utils.pattern_match(args.tasks, task_manager.all_tasks)
        results = {}
        for task_name in task_names:
            logger.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.lm_eval_batch_size, task_manager=task_manager)['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            results[task_name] = acc
            logger.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        logger.info(metric_vals)


if __name__ == '__main__':
    main()