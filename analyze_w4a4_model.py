#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append('/workspace/FlatQuant')

import transformers
import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.flat_utils as flat_utils
import gptq_utils

def analyze_w4a4_model():
    """Analyze the working W4A4 model to understand its structure"""
    print("🔍 ANALYZING WORKING W4A4 MODEL STRUCTURE")
    print("="*60)
    
    # Load the working model
    class Args:
        def __init__(self):
            self.model = "./modelzoo/llama-3.2-1b"
            self.hf_token = None
            self.seed = 0
            # W4A4KV4 settings
            self.w_bits = 4
            self.a_bits = 4
            self.k_bits = 4
            self.v_bits = 4
            self.k_asym = True
            self.v_asym = True
            self.k_groupsize = 128
            self.v_groupsize = 128
            self.w_asym = False
            self.a_asym = False
            self.w_groupsize = -1
            self.a_groupsize = -1
            # FlatQuant settings
            self.quantize = True
            self.lwc = True
            self.lac = True
            self.cali_trans = True
            self.add_diag = True
            self.reload_matrix = True
            self.matrix_path = "./outputs/llama-3.2-1b/w4a4/exp"
            self.resume = False
            self.save_matrix = False
            self.gptq = False
            self.distribute_model = False
            # Additional attributes from log file
            self.direct_inv = False
            self.flat_lr = 0.005
            self.epochs = 15
            self.epoch = 15
            self.q_bits = 16
            self.q_asym = False
            self.q_groupsize = -1
            self.diag_alpha = 0.3
            self.diag_init = 'sq_style'
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
            self.separate_vtrans = False
            self.tasks = ['piqa', 'hellaswag', 'arc_easy', 'arc_challenge', 'winogrande', 'lambada_openai']
            self.warmup = False
    
    args = Args()
    utils.seed_everything(seed=args.seed)
    
    print("📂 Loading original model...")
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    print("🔧 Applying FlatQuant transformations...")
    model = apply_flatquant_to_model(args, model)
    
    print("📥 Loading pre-trained matrices...")
    flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
    flat_utils.reparameterize_model(model)
    
    print("⚙️ Applying weight quantization...")
    quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
    
    model.to(utils.DEV)
    
    print(f"\n📊 MODEL ANALYSIS:")
    print(f"Total layers: {len(list(model.named_parameters()))}")
    
    # Analyze the structure
    quantized_layers = 0
    embedding_layers = 0
    other_layers = 0
    
    print(f"\n🔍 LAYER-BY-LAYER ANALYSIS:")
    
    for name, param in model.named_parameters():
        param_size_mb = param.numel() * param.element_size() / (1024**2)
        
        if 'embed_tokens' in name:
            embedding_layers += 1
            print(f"📝 EMBEDDING: {name}")
            print(f"   Shape: {param.shape}")
            print(f"   Size: {param_size_mb:.2f} MB")
            print(f"   Dtype: {param.dtype}")
            
        elif any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
            quantized_layers += 1
            print(f"⚙️ QUANTIZED: {name}")
            print(f"   Shape: {param.shape}")
            print(f"   Size: {param_size_mb:.2f} MB")
            print(f"   Dtype: {param.dtype}")
            
            # Check if this layer has a quantizer
            layer_name = name.replace('.weight', '')
            if layer_name in quantizers:
                q = quantizers[layer_name]
                print(f"   Quantizer: scale shape {q.scale.shape}, zero shape {q.zero.shape}")
                print(f"   Quantizer maxq: {q.maxq}")
            else:
                print(f"   ⚠️ No quantizer found for {layer_name}")
                
        else:
            other_layers += 1
            print(f"🔧 OTHER: {name}")
            print(f"   Shape: {param.shape}")
            print(f"   Size: {param_size_mb:.2f} MB")
            print(f"   Dtype: {param.dtype}")
    
    print(f"\n📈 SUMMARY:")
    print(f"  Embedding layers: {embedding_layers}")
    print(f"  Quantized layers: {quantized_layers}")
    print(f"  Other layers: {other_layers}")
    print(f"  Total quantizers: {len(quantizers)}")
    
    # Analyze quantizers
    print(f"\n🎯 QUANTIZER ANALYSIS:")
    total_quantized_params = 0
    for layer_name, q in quantizers.items():
        param_count = q.scale.numel()
        total_quantized_params += param_count
        print(f"  {layer_name}:")
        print(f"    Scale shape: {q.scale.shape}")
        print(f"    Zero shape: {q.zero.shape}")
        print(f"    Maxq: {q.maxq}")
        print(f"    Parameters: {param_count:,}")
    
    print(f"  Total quantized parameters: {total_quantized_params:,}")
    
    # Check how weights are actually stored after quantization
    print(f"\n🔬 WEIGHT STORAGE ANALYSIS:")
    for name, param in model.named_parameters():
        if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            print(f"  {name}:")
            print(f"    Current weight range: [{param.min().item():.3f}, {param.max().item():.3f}]")
            print(f"    Unique values: {len(torch.unique(param))}")
            # Check if values look quantized (should be integers for 4-bit)
            if len(torch.unique(param)) <= 16:
                print(f"    ✅ Appears quantized (≤16 unique values)")
            else:
                print(f"    ⚠️ May not be quantized ({len(torch.unique(param))} unique values)")
            break  # Just check first layer
    
    # Save analysis results
    analysis_data = {
        'quantizers': quantizers,
        'model_state': {name: param.detach().cpu() for name, param in model.named_parameters()},
        'layer_counts': {
            'embedding': embedding_layers,
            'quantized': quantized_layers, 
            'other': other_layers
        }
    }
    
    torch.save(analysis_data, '/workspace/FlatQuant/w4a4_analysis.pth')
    print(f"\n💾 Analysis saved to w4a4_analysis.pth")
    
    return model, quantizers

if __name__ == "__main__":
    model, quantizers = analyze_w4a4_model()
    print(f"\n✅ Analysis completed!")