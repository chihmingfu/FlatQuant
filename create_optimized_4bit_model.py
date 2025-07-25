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

def pack_4bit_optimized(indices_tensor):
    """Pack 4-bit indices efficiently: 2 values per uint8"""
    # Ensure indices are in valid 4-bit range [0, 15]
    # Convert from signed [-8, 7] to unsigned [0, 15]
    unsigned_indices = (indices_tensor + 8).clamp(0, 15).to(torch.uint8)
    
    # Flatten and pad if necessary
    flat_indices = unsigned_indices.flatten()
    if flat_indices.numel() % 2 == 1:
        # Pad with 0 if odd number of elements
        flat_indices = torch.cat([flat_indices, torch.tensor([0], dtype=torch.uint8)])
    
    # Pack pairs: lower 4 bits + upper 4 bits
    pairs = flat_indices.view(-1, 2)
    packed = pairs[:, 0] + (pairs[:, 1] << 4)
    
    return packed, unsigned_indices.shape

def create_optimized_4bit_model():
    """Create optimized 4-bit model with true 4:1 compression"""
    print("🔧 CREATING OPTIMIZED 4-BIT MODEL (TARGET: 4:1 COMPRESSION)")
    print("="*70)
    
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
    
    print(f"✅ Applied FlatQuant W4 quantization with {len(quantizers)} quantizers")
    
    # Optimized compression with bit packing
    compressed_data = {}
    total_original_size = 0
    total_compressed_size = 0
    
    print(f"\n📦 Optimized compression with 4-bit packing:")
    
    for name, param in model.named_parameters():
        param_data = param.data.cpu()
        layer_name = None
        
        # Find corresponding quantizer
        for q_name in quantizers.keys():
            if q_name in name:
                layer_name = q_name
                break
        
        if layer_name and layer_name in quantizers:
            quantizer = quantizers[layer_name]
            print(f"  Processing {name}")
            
            # Extract quantization parameters
            scale = quantizer.scale
            maxq = quantizer.maxq
            
            # Apply FlatQuant's exact quantization
            param_gpu = param_data.to(device)
            scale_gpu = scale.to(param_gpu.device)
            
            from flatquant.quant_utils import round_ste
            w_div_scale = param_gpu / scale_gpu
            min_bound = torch.tensor(-(maxq + 1), device=param_gpu.device, dtype=param_gpu.dtype)
            max_bound = torch.tensor(maxq, device=param_gpu.device, dtype=param_gpu.dtype)
            w_quantized = torch.clamp(round_ste(w_div_scale), min_bound, max_bound)
            
            # Optimized bit packing
            packed_indices, original_shape = pack_4bit_optimized(w_quantized.cpu())
            
            # Store with minimal metadata
            compressed_data[f"{name}_packed"] = packed_indices
            compressed_data[f"{name}_scale"] = scale.cpu().to(torch.float16)  # Use FP16 for scales
            compressed_data[f"{name}_shape"] = torch.tensor(original_shape, dtype=torch.int32)  # Use INT32 instead of INT64
            
            # Calculate compression
            original_size = param_data.numel() * 2  # BF16 = 2 bytes
            packed_size = packed_indices.numel() * 1  # 1 byte per packed pair
            scale_size = scale.numel() * 2  # FP16 scales
            shape_size = len(original_shape) * 4  # INT32 shape
            compressed_size = packed_size + scale_size + shape_size
            
            total_original_size += original_size
            total_compressed_size += compressed_size
            
            compression = original_size / compressed_size
            print(f"    {original_size} -> {compressed_size} bytes ({compression:.2f}x)")
            
        else:
            # Keep normalization layers in BF16
            print(f"  Keeping {name} in BF16")
            compressed_data[name] = param_data.to(torch.float16)  # Use FP16 for normalization too
            
            original_size = param_data.numel() * 2  # BF16
            compressed_size = param_data.numel() * 2  # FP16 (same size for small tensors)
            total_original_size += original_size
            total_compressed_size += compressed_size
    
    # Calculate overall compression
    overall_compression = total_original_size / total_compressed_size
    compressed_size_gb = total_compressed_size / (1024**3)
    
    print(f"\n📊 Optimized Compression Results:")
    print(f"Original size: {total_original_size/(1024**3):.2f} GB")
    print(f"Compressed size: {compressed_size_gb:.2f} GB")
    print(f"Overall compression: {overall_compression:.2f}x")
    
    # Save the optimized model
    output_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-optimized-4bit/model_optimized_4bit.safetensors"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n💾 Saving to: {output_path}")
    save_file(compressed_data, output_path)
    
    actual_size = os.path.getsize(output_path)
    actual_size_gb = actual_size / (1024**3)
    print(f"Actual file size: {actual_size_gb:.2f} GB")
    
    # Check compression target
    target_size = 2.30 / 4  # 0.575 GB for 4:1 compression
    actual_compression = 2.30 / actual_size_gb
    
    print(f"Target size: {target_size:.2f} GB (4:1 compression)")
    print(f"Actual compression: {actual_compression:.2f}x")
    
    if actual_compression >= 3.8:  # Allow some tolerance
        print("✅ Compression target achieved!")
        status = "SUCCESS"
    elif actual_compression >= 3.0:
        print("⚠️  Close to compression target")
        status = "CLOSE"
    else:
        print("❌ Compression target not reached")
        status = "NEEDS_WORK"
    
    # Save quantizers for reconstruction
    quantizer_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-optimized-4bit/quantizers.pth"
    torch.save(quantizers, quantizer_path)
    print(f"💾 Saved quantizers to: {quantizer_path}")
    
    return output_path, quantizer_path, actual_size_gb, actual_compression, status

if __name__ == "__main__":
    model_path, quantizer_path, file_size, compression, status = create_optimized_4bit_model()
    
    print(f"\n" + "="*70)
    print("✅ OPTIMIZED 4-BIT MODEL CREATED")
    print("="*70)
    print(f"📁 Model: {model_path}")
    print(f"📁 Quantizers: {quantizer_path}")
    print(f"📏 Size: {file_size:.2f} GB")
    print(f"📊 Compression: {compression:.2f}x")
    print(f"🎯 Status: {status}")
    print(f"💡 Uses optimized 4-bit packing + FP16 metadata")
    print(f"📝 Ready for testing with enhanced compression")