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
import flatquant.quant_utils as quant_utils
import gptq_utils

def extract_quantized_indices():
    """Extract 4-bit quantized indices from the working W4A4 model"""
    print("🔬 EXTRACTING QUANTIZED INDICES FROM W4A4 MODEL")
    print("="*60)
    
    # Load analysis data
    if os.path.exists('/workspace/FlatQuant/w4a4_analysis.pth'):
        print("📥 Loading previous analysis...")
        analysis_data = torch.load('/workspace/FlatQuant/w4a4_analysis.pth', weights_only=False)
        quantizers = analysis_data['quantizers']
        print(f"✅ Loaded {len(quantizers)} quantizers from analysis")
    else:
        print("⚠️ No analysis found, need to run analysis first")
        return
    
    # Load the working model to get current weights
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
            # Complete attributes
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
    
    print("📂 Loading working model...")
    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    print("🔧 Applying FlatQuant transformations...")
    model = apply_flatquant_to_model(args, model)
    flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
    flat_utils.reparameterize_model(model)
    
    print("⚙️ Applying weight quantization...")
    quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
    
    model.to(utils.DEV)
    
    print(f"\n🎯 EXTRACTING 4-BIT INDICES:")
    
    extracted_data = {
        'embedding': None,
        'quantized_layers': {},
        'flatquant_matrices': {},
        'metadata': {
            'model_path': args.model,
            'quantization_config': {
                'w_bits': args.w_bits,
                'a_bits': args.a_bits,
                'k_bits': args.k_bits,
                'v_bits': args.v_bits,
                'k_asym': args.k_asym,
                'v_asym': args.v_asym,
                'k_groupsize': args.k_groupsize,
                'v_groupsize': args.v_groupsize
            }
        }
    }
    
    # Extract embedding (unquantized)
    print("📝 Extracting embedding layer...")
    for name, param in model.named_parameters():
        if 'embed_tokens' in name:
            extracted_data['embedding'] = {
                'weight': param.detach().cpu(),
                'shape': param.shape,
                'dtype': param.dtype
            }
            embedding_size_mb = param.numel() * param.element_size() / (1024**2)
            print(f"   Embedding: {param.shape} ({embedding_size_mb:.1f} MB)")
            break
    
    # Extract quantized weights and reverse-engineer indices
    print("\n🔬 Reverse-engineering 4-bit indices from quantized weights...")
    total_compressed_size = 0
    
    for layer_name, quantizer in quantizers.items():
        print(f"\n🔍 Processing {layer_name}:")
        
        # Get the current quantized weight
        # Find corresponding weight in model
        weight_param = None
        for name, param in model.named_parameters():
            if name == f"{layer_name}.weight":
                weight_param = param
                break
        
        if weight_param is None:
            print(f"   ⚠️ Weight not found for {layer_name}")
            continue
            
        # Reverse quantization to get indices
        # Current weight = scale * indices + zero
        # So: indices = (weight - zero) / scale
        scale = quantizer.scale.to(weight_param.device)
        zero = quantizer.zero.to(weight_param.device)
        
        # Reverse the quantization
        weight_shifted = weight_param - zero
        indices_float = weight_shifted / scale
        
        # Round to nearest integer (these should be the original 4-bit indices)
        indices = torch.round(indices_float).to(torch.int8)
        
        # Validate indices are in 4-bit range [-8, 7] (for signed) or [0, 15] (for unsigned)
        min_val = indices.min().item()
        max_val = indices.max().item()
        unique_vals = len(torch.unique(indices))
        
        print(f"   Shape: {indices.shape}")
        print(f"   Index range: [{min_val}, {max_val}]")
        print(f"   Unique values: {unique_vals}")
        print(f"   Quantizer maxq: {quantizer.maxq}")
        
        # Check if indices are valid 4-bit
        if unique_vals <= 16 and min_val >= -8 and max_val <= 7:
            print(f"   ✅ Valid 4-bit indices")
            
            # Calculate compressed size (2 indices per byte)
            num_elements = indices.numel()
            compressed_bytes = (num_elements + 1) // 2  # Round up for odd numbers
            compressed_mb = compressed_bytes / (1024**2)
            original_mb = weight_param.numel() * weight_param.element_size() / (1024**2)
            compression_ratio = original_mb / compressed_mb
            
            print(f"   Original size: {original_mb:.2f} MB")
            print(f"   Compressed size: {compressed_mb:.2f} MB")
            print(f"   Compression ratio: {compression_ratio:.1f}x")
            
            total_compressed_size += compressed_mb
            
            # Store extracted data
            extracted_data['quantized_layers'][layer_name] = {
                'indices': indices.cpu(),
                'scale': quantizer.scale.cpu(),
                'zero': quantizer.zero.cpu(),
                'shape': indices.shape,
                'maxq': quantizer.maxq,
                'original_size_mb': original_mb,
                'compressed_size_mb': compressed_mb
            }
            
        else:
            print(f"   ❌ Invalid 4-bit indices - may need different approach")
            # For debugging, let's also look at the raw weight values
            weight_unique = len(torch.unique(weight_param))
            print(f"   Weight unique values: {weight_unique}")
            print(f"   Weight range: [{weight_param.min().item():.3f}, {weight_param.max().item():.3f}]")
    
    # Load FlatQuant matrices
    print(f"\n📊 Loading FlatQuant transformation matrices...")
    matrix_path = os.path.join(args.matrix_path, 'flat_matrices.pth')
    if os.path.exists(matrix_path):
        flat_matrices = torch.load(matrix_path, map_location='cpu', weights_only=False)
        extracted_data['flatquant_matrices'] = flat_matrices
        
        # Calculate matrix size carefully
        matrix_size_mb = 0
        for k, v in flat_matrices.items():
            if hasattr(v, 'numel'):
                matrix_size_mb += v.numel() * v.element_size()
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if hasattr(v2, 'numel'):
                        matrix_size_mb += v2.numel() * v2.element_size()
        matrix_size_mb = matrix_size_mb / (1024**2)
        print(f"   FlatQuant matrices: {matrix_size_mb:.2f} MB")
    else:
        print(f"   ⚠️ FlatQuant matrices not found at {matrix_path}")
        matrix_size_mb = 0
    
    # Summary
    embedding_size_mb = extracted_data['embedding']['weight'].numel() * extracted_data['embedding']['weight'].element_size() / (1024**2)
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   Embedding: {embedding_size_mb:.1f} MB (uncompressed)")
    print(f"   Quantized weights: {total_compressed_size:.1f} MB (4-bit compressed)")
    print(f"   FlatQuant matrices: {matrix_size_mb:.2f} MB")
    print(f"   Total estimated size: {embedding_size_mb + total_compressed_size + matrix_size_mb:.1f} MB")
    print(f"   Quantized layers extracted: {len(extracted_data['quantized_layers'])}")
    
    # Save extracted data
    output_path = '/workspace/FlatQuant/extracted_quantized_data.pth'
    torch.save(extracted_data, output_path)
    print(f"\n💾 Extracted data saved to: {output_path}")
    
    return extracted_data

if __name__ == "__main__":
    extracted_data = extract_quantized_indices()
    print(f"\n✅ Phase 2 completed!")