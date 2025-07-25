#!/usr/bin/env python3

import torch
import sys
import os
from safetensors.torch import save_file

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')
sys.path.append('/workspace/FlatQuant/flatquant')

import flatquant.model_utils as model_utils
import flatquant.utils as utils
import gptq_utils

def save_runtime_w4_compressed():
    """Save FlatQuant's runtime W4 weights in proper 4-bit compressed format"""
    print("🔧 SAVING RUNTIME W4 WEIGHTS IN 4-BIT COMPRESSED FORMAT")
    print("="*60)
    
    device = 'cuda'
    model_path = "./modelzoo/llama-3.2-1b"
    
    # Load model and apply FlatQuant W4
    class Args:
        def __init__(self):
            self.model = model_path
            self.hf_token = None
    
    args = Args()
    model, tokenizer = model_utils.get_model(args.model, args.hf_token)
    
    class FlatQuantArgs:
        def __init__(self):
            self.w_bits = 4
            self.w_asym = False
            self.w_groupsize = -1
            self.a_bits = 16
            self.k_bits = 16
            self.v_bits = 16
            self.gptq = False
            self.gptq_mse = False
    
    flatquant_args = FlatQuantArgs()
    model = model.to(device)
    quantizers = gptq_utils.rtn_fwrd(model, device, flatquant_args)
    model.eval()
    
    print("✅ Applied FlatQuant W4 quantization")
    
    # Extract and compress weights
    compressed_weights = {}
    original_total_size = 0
    compressed_total_size = 0
    
    quantize_patterns = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj', 'embed_tokens'
    ]
    
    for name, param in model.named_parameters():
        param_data = param.data.cpu()
        original_size = param_data.numel() * 2  # BF16 = 2 bytes
        original_total_size += original_size
        
        # Check if this parameter should be quantized
        should_quantize = any(pattern in name for pattern in quantize_patterns)
        
        if should_quantize:
            # The weights are already quantized by FlatQuant, now pack them to 4-bit
            print(f"Compressing {name} (shape: {param_data.shape})")
            
            # Clamp to 4-bit range [-8, 7] and convert to int
            clamped = torch.clamp(param_data, -8, 7).round().to(torch.int8)
            
            # Pack two 4-bit values into one uint8
            flat_weights = clamped.flatten()
            
            # Pad if odd number of elements
            if flat_weights.numel() % 2 == 1:
                flat_weights = torch.cat([flat_weights, torch.tensor([0], dtype=torch.int8)])
            
            # Reshape to pairs and pack
            pairs = flat_weights.view(-1, 2)
            
            # Map [-8,7] to [0,15] for unsigned storage
            val1 = (pairs[:, 0] + 8).to(torch.uint8)
            val2 = (pairs[:, 1] + 8).to(torch.uint8)
            
            # Pack: lower 4 bits + upper 4 bits
            packed = val1 + (val2 << 4)
            
            # Store compressed data with metadata
            compressed_weights[f"{name}_packed"] = packed
            compressed_weights[f"{name}_shape"] = torch.tensor(param_data.shape, dtype=torch.int64)
            compressed_weights[f"{name}_original_numel"] = torch.tensor([param_data.numel()], dtype=torch.int64)
            
            compressed_size = packed.numel() + param_data.ndim * 8 + 8  # packed data + shape + numel
            compressed_total_size += compressed_size
            
            compression_ratio = original_size / compressed_size
            print(f"  {name}: {original_size} -> {compressed_size} bytes ({compression_ratio:.1f}x compression)")
            
        else:
            # Keep normalization layers in BF16
            print(f"Keeping {name} in BF16 (shape: {param_data.shape})")
            compressed_weights[name] = param_data
            compressed_total_size += original_size
    
    print(f"\n📊 Compression Summary:")
    print(f"Original total size: {original_total_size/1024/1024/1024:.2f} GB")
    print(f"Compressed total size: {compressed_total_size/1024/1024/1024:.2f} GB")
    overall_compression = original_total_size / compressed_total_size
    print(f"Overall compression: {overall_compression:.2f}x")
    
    # Save compressed model
    output_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-compressed/model_runtime_w4_compressed.safetensors"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n💾 Saving to: {output_path}")
    save_file(compressed_weights, output_path)
    
    actual_size = os.path.getsize(output_path)
    print(f"Actual file size: {actual_size/1024/1024/1024:.2f} GB")
    
    target_size = 2.30 / 4  # 0.575 GB
    size_ratio = actual_size / (target_size * 1024**3)
    print(f"Target size: {target_size:.2f} GB")
    print(f"Size ratio: {size_ratio:.2f}x (target is 1.0x)")
    
    if size_ratio <= 1.1:  # Within 10% of target
        print("✅ Compression target achieved!")
    else:
        print("❌ Compression target missed")
    
    return output_path, actual_size

if __name__ == "__main__":
    output_path, file_size = save_runtime_w4_compressed()
    print(f"\n✅ Runtime W4 weights saved in compressed format")
    print(f"📝 File: {output_path}")
    print(f"📏 Size: {file_size/1024/1024/1024:.2f} GB")