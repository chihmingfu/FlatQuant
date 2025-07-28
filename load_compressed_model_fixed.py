#!/usr/bin/env python3

import torch
import sys
import os
import json
from safetensors import safe_open

sys.path.append('/workspace/FlatQuant')

import transformers
import flatquant.utils as utils
import flatquant.model_utils as model_utils
import flatquant.flat_utils as flat_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils

def unpack_4bit_indices(packed_bytes, original_shape, original_numel):
    """Unpack 4-bit indices from bytes"""
    low_nibbles = packed_bytes & 0x0F
    high_nibbles = (packed_bytes >> 4) & 0x0F
    unpacked = torch.stack([low_nibbles, high_nibbles], dim=1).flatten()
    unpacked = unpacked[:original_numel]
    signed_indices = (unpacked.to(torch.int8) - 7).clamp(-7, 7)
    restored = signed_indices.view(original_shape)
    return restored

def reconstruct_weight_from_quantizer(indices, scale, zero):
    """Reconstruct weight from quantized indices"""
    indices_gpu = indices.to(scale.device).to(scale.dtype)
    reconstructed_weight = scale * indices_gpu + zero
    return reconstructed_weight

def load_compressed_model_fixed(compressed_model_path, device='cuda'):
    """Load compressed model using ORIGINAL FlatQuant matrices from working directory"""
    print("📂 LOADING COMPRESSED MODEL WITH ORIGINAL FLATQUANT MATRICES")
    print("="*70)
    
    # Load compressed data
    compressed_data = {}
    metadata = {}
    
    with safe_open(compressed_model_path, framework="pt") as f:
        for key in f.keys():
            compressed_data[key] = f.get_tensor(key)
        
        safetensors_metadata = f.metadata()
        if safetensors_metadata:
            for key, value in safetensors_metadata.items():
                try:
                    metadata[key] = json.loads(value)
                except:
                    metadata[key] = value
    
    model_metadata = metadata.get('model_metadata', {})
    compression_info = model_metadata.get('compression_info', {})
    
    print(f"📊 Compressed model: {compression_info.get('compressed_size_mb', 0):.1f} MB")
    
    # Setup args to use ORIGINAL FlatQuant matrices
    class Args:
        def __init__(self):
            self.model = "./modelzoo/llama-3.2-1b"
            self.hf_token = None
            self.seed = 0
            # Quantization config  
            config = model_metadata.get('quantization_config', {})
            self.w_bits = config.get('w_bits', 4)
            self.a_bits = config.get('a_bits', 4) 
            self.k_bits = config.get('k_bits', 4)
            self.v_bits = config.get('v_bits', 4)
            self.k_asym = config.get('k_asym', True)
            self.v_asym = config.get('v_asym', True)
            self.k_groupsize = config.get('k_groupsize', 128)
            self.v_groupsize = config.get('v_groupsize', 128)
            self.w_asym = config.get('w_asym', False)
            self.a_asym = config.get('a_asym', False)
            self.w_groupsize = -1
            self.a_groupsize = -1
            # FlatQuant settings - USE ORIGINAL MATRICES
            self.quantize = True
            self.lwc = True
            self.lac = True
            self.cali_trans = True
            self.add_diag = True
            self.reload_matrix = True
            self.matrix_path = "./outputs/llama-3.2-1b/w4a4/exp"  # USE ORIGINAL PATH
            self.resume = False
            self.save_matrix = False
            self.gptq = False
            self.distribute_model = False
            # Required attributes
            self.direct_inv = False
            self.flat_lr = 0.005
            self.epochs = 15
            self.epoch = 15
            self.diag_alpha = 0.3
            self.diag_init = 'sq_style'
            self.separate_vtrans = False
            self.q_bits = 16
            self.q_asym = False
            self.q_groupsize = -1
            self.deactive_amp = False
            self.act_order = False
            self.cache_dir = './outputs/.cache'
            self.cali_bsz = 4
            self.cali_dataset = 'wikitext2'
            self.exp_dir = './outputs/llama-3.2-1b/w4a4/exp'
            self.exp_name = 'exp'
            self.gptq_mse = False
            self.lm_eval = False
            self.lm_eval_batch_size = 128
            self.model_name = 'llama-3.2-1b'
            self.nsamples = 128
            self.output_dir = './outputs'
            self.percdamp = 0.01
            self.tasks = ['piqa', 'hellaswag', 'arc_easy', 'arc_challenge', 'winogrande', 'lambada_openai']
            self.warmup = False
    
    args = Args()
    utils.seed_everything(seed=args.seed)
    
    print(f"📂 Loading base model structure...")
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    print(f"🔧 Applying FlatQuant transformations...")
    model = apply_flatquant_to_model(args, model)
    
    print(f"📥 Loading ORIGINAL FlatQuant matrices from working directory...")
    flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
    print(f"   ✅ Loaded FlatQuant matrices from {args.matrix_path}")
    
    print(f"🔄 Reparameterizing model...")
    flat_utils.reparameterize_model(model)
    
    print(f"🔧 Reconstructing quantized weights from compressed data...")
    
    # Build lookup table for layer names to weight parameters
    layer_lookup = {}
    for name, param in model.named_parameters():
        if '.linear.weight' in name:
            layer_name = name.replace('.weight', '')
            layer_lookup[layer_name] = param
    
    # Reconstruct each quantized layer
    replaced_layers = 0
    for layer_name in layer_lookup.keys():
        packed_key = f"{layer_name}_packed_indices"
        metadata_key = f"{layer_name}_pack_metadata"
        scale_key = f"{layer_name}_scale"
        zero_key = f"{layer_name}_zero"
        
        if packed_key in compressed_data:
            # Load compressed data
            packed_indices = compressed_data[packed_key]
            pack_metadata = metadata[metadata_key]
            scale = compressed_data[scale_key].to(device)
            zero = compressed_data[zero_key].to(device)
            
            # Unpack and reconstruct
            original_shape = pack_metadata['original_shape']
            original_numel = pack_metadata['original_numel']
            unpacked_indices = unpack_4bit_indices(packed_indices, original_shape, original_numel)
            reconstructed_weight = reconstruct_weight_from_quantizer(unpacked_indices, scale, zero)
            
            # Replace weight directly
            weight_param = layer_lookup[layer_name]
            weight_param.data.copy_(reconstructed_weight.to(weight_param.dtype))
            replaced_layers += 1
    
    print(f"   ✅ Reconstructed {replaced_layers} quantized layers")
    
    # Load embedding layer
    print(f"📝 Loading embedding layer...")
    if 'embedding_weight' in compressed_data:
        embedding_weight = compressed_data['embedding_weight']
        for name, param in model.named_parameters():
            if 'embed_tokens' in name:
                param.data.copy_(embedding_weight.to(param.device))
                break
        print(f"   ✅ Loaded embedding layer")
    
    model.to(device)
    print(f"✅ Model loading completed!")
    
    return model, tokenizer

def test_compressed_model_fixed():
    """Test the compressed model with corrected FlatQuant matrix loading"""
    print("🎯 TESTING COMPRESSED MODEL (FIXED VERSION)")
    print("="*55)
    
    device = 'cuda'
    compressed_model_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit-compressed/model_true_4bit_compressed.safetensors"
    
    # Load model
    model, tokenizer = load_compressed_model_fixed(compressed_model_path, device)
    
    # Generation test
    print(f"\n🧑 Generation Test:")
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated = model.generate(
            inputs.input_ids,
            max_new_tokens=15,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
        )
    
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Input: {test_input}")
    print(f"Output: {output_text}")
    
    # Check if generation is reasonable
    if "Paris" in output_text:
        print(f"✅ Generation test PASSED - output contains 'Paris'")
        generation_success = True
    else:
        print(f"⚠️ Generation test may have issues - no 'Paris' in output")
        generation_success = False
    
    # Only run perplexity test if generation looks good
    if generation_success:
        print(f"\n📚 Evaluating on WikiText-2...")
        
        class Args:
            def __init__(self):
                self.model = "./modelzoo/llama-3.2-1b"
                self.hf_token = None
                self.seed = 0
                self.cali_dataset = "wikitext2"
                self.cache_dir = './outputs/.cache'
        
        args = Args()
        
        testloader = data_utils.get_loaders(
            args,
            "wikitext2",
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )
        
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        print(f"WikiText-2 perplexity: {dataset_ppl:.3f}")
        
        # Compare with target
        target_ppl = 11.973
        print(f"\n📊 FINAL RESULTS:")
        print(f"  Target PPL (working model): {target_ppl:.3f}")
        print(f"  Compressed model PPL: {dataset_ppl:.3f}")
        print(f"  Difference: {abs(dataset_ppl - target_ppl):.3f}")
        
        if abs(dataset_ppl - target_ppl) < 0.1:
            print(f"  ✅ EXCELLENT: PPL matches target very closely!")
        elif abs(dataset_ppl - target_ppl) < 1.0:
            print(f"  ✅ GOOD: PPL is very close to target")
        else:
            print(f"  ⚠️ PPL differs from target")
        
        return dataset_ppl
    else:
        print(f"\n⚠️ Skipping perplexity test due to generation issues")
        return None

if __name__ == "__main__":
    ppl = test_compressed_model_fixed()
    if ppl is not None:
        print(f"\n✅ Testing completed! PPL: {ppl:.3f}")
    else:
        print(f"\n⚠️ Testing completed with issues")