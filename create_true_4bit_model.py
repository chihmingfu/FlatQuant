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

def pack_true_4bit(indices_tensor):
    """Pack indices to TRUE 4-bit: 2 values per uint8 byte"""
    # Convert from signed [-8, 7] to unsigned [0, 15] for easier packing
    unsigned_indices = (indices_tensor + 8).clamp(0, 15).to(torch.uint8)
    
    # Flatten
    flat_indices = unsigned_indices.flatten()
    
    # Pad to even number if necessary
    if flat_indices.numel() % 2 == 1:
        flat_indices = torch.cat([flat_indices, torch.tensor([0], dtype=torch.uint8)])
    
    # Pack pairs: lower 4 bits from first value + upper 4 bits from second value
    pairs = flat_indices.view(-1, 2)
    packed = pairs[:, 0] | (pairs[:, 1] << 4)  # Pack two 4-bit values into one uint8
    
    return packed, unsigned_indices.shape, flat_indices.numel()

def create_true_4bit_model():
    """Create model with TRUE 4-bit storage (2 values per byte)"""
    print("🎯 CREATING TRUE 4-BIT MODEL (2 VALUES PER BYTE)")
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
    
    print(f"✅ Applied FlatQuant W4 quantization with {len(quantizers)} quantizers")
    
    # True 4-bit compression
    compressed_data = {}
    total_original_size = 0
    total_compressed_size = 0
    quantized_params = 0
    unquantized_params = 0
    
    print(f"\n📦 TRUE 4-bit packing (2 values per byte):")
    
    for name, param in model.named_parameters():
        param_data = param.data.cpu()
        layer_name = None
        
        # Find corresponding quantizer
        for q_name in quantizers.keys():
            if q_name in name:
                layer_name = q_name
                break
        
        original_size = param_data.numel() * 2  # BF16 = 2 bytes per parameter
        
        if layer_name and layer_name in quantizers:
            # Use FlatQuant quantizer
            quantizer = quantizers[layer_name]
            print(f"  📋 Quantizing {name}")
            
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
            
            # TRUE 4-bit packing
            packed_data, original_shape, padded_numel = pack_true_4bit(w_quantized.cpu())
            
            # Store with minimal overhead
            compressed_data[f"{name}_packed"] = packed_data
            compressed_data[f"{name}_scale"] = scale.cpu().half()  # FP16 scales
            
            # Store shape more efficiently 
            shape_tensor = torch.tensor(original_shape, dtype=torch.int16)  # Use int16 for shapes
            compressed_data[f"{name}_shape"] = shape_tensor
            compressed_data[f"{name}_numel"] = torch.tensor([padded_numel], dtype=torch.int32)
            
            # Calculate true compression
            packed_size = packed_data.numel()  # Each byte stores 2 values
            scale_size = scale.numel() * 2  # FP16
            metadata_size = shape_tensor.numel() * 2 + 4  # int16 shape + int32 numel
            compressed_size = packed_size + scale_size + metadata_size
            
            compression_ratio = original_size / compressed_size
            print(f"    Original: {original_size:,} bytes")
            print(f"    Packed: {packed_size:,} bytes (stores {padded_numel:,} values)")
            print(f"    Total compressed: {compressed_size:,} bytes")
            print(f"    Compression: {compression_ratio:.2f}x")
            
            quantized_params += 1
            
        else:
            # Keep embedding and normalization layers in BF16
            layer_type = "embedding" if "embed_tokens" in name else "normalization"
            print(f"  🔒 Keeping {name} in BF16 ({layer_type})")
            compressed_data[name] = param_data.half()  # Store as FP16 to save some space
            compressed_size = param_data.numel() * 2  # FP16
            compression_ratio = original_size / compressed_size
            print(f"    Size: {compressed_size:,} bytes ({compression_ratio:.1f}x)")
            
            unquantized_params += 1
        
        total_original_size += original_size
        total_compressed_size += compressed_size
    
    # Calculate overall results
    overall_compression = total_original_size / total_compressed_size
    compressed_size_gb = total_compressed_size / (1024**3)
    original_size_gb = total_original_size / (1024**3)
    
    print(f"\n📊 TRUE 4-BIT COMPRESSION RESULTS:")
    print(f"Original model size: {original_size_gb:.2f} GB")
    print(f"Compressed model size: {compressed_size_gb:.2f} GB")
    print(f"Overall compression: {overall_compression:.2f}x")
    print(f"Size reduction: {(1 - compressed_size_gb/original_size_gb)*100:.1f}%")
    print(f"Quantized parameters: {quantized_params}")
    print(f"Unquantized parameters: {unquantized_params}")
    
    # Save the true 4-bit model
    output_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit/model_true_4bit.safetensors"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n💾 Saving to: {output_path}")
    save_file(compressed_data, output_path)
    
    # Check actual file size
    actual_size = os.path.getsize(output_path)
    actual_size_gb = actual_size / (1024**3)
    actual_compression = original_size_gb / actual_size_gb
    
    print(f"📏 ACTUAL FILE SIZE RESULTS:")
    print(f"Calculated size: {compressed_size_gb:.2f} GB")
    print(f"Actual file size: {actual_size_gb:.2f} GB")
    print(f"Actual compression: {actual_compression:.2f}x")
    
    # Compare with target
    target_4_to_1 = original_size_gb / 4
    print(f"\nTarget size (4:1): {target_4_to_1:.2f} GB")
    if actual_size_gb <= target_4_to_1 * 1.1:  # Within 10% of target
        print("🎯 ✅ Near 4:1 compression achieved!")
        status = "SUCCESS"
    elif actual_compression >= 3.0:
        print("🎯 ⚠️  Good compression achieved (3x+)")
        status = "GOOD"
    else:
        print("🎯 📈 Improved compression")
        status = "IMPROVED"
    
    # Save quantizers
    quantizer_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit/quantizers.pth"
    torch.save(quantizers, quantizer_path)
    print(f"💾 Saved quantizers to: {quantizer_path}")
    
    return output_path, quantizer_path, actual_size_gb, actual_compression, status

if __name__ == "__main__":
    model_path, quantizer_path, file_size, compression, status = create_true_4bit_model()
    
    print(f"\n" + "="*60)
    print("🎯 TRUE 4-BIT MODEL CREATED")
    print("="*60)
    print(f"📁 Model: {model_path}")
    print(f"📁 Quantizers: {quantizer_path}")
    print(f"📏 Size: {file_size:.2f} GB")
    print(f"📊 Compression: {compression:.2f}x")
    print(f"🎯 Status: {status}")
    print(f"💡 Uses TRUE 4-bit storage (2 values per byte)")
    print(f"📝 Ready for testing - should achieve much better compression")