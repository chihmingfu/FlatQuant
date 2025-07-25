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

def create_proper_4bit_model():
    """Create 4-bit model using FlatQuant's actual quantization parameters"""
    print("🔧 CREATING PROPER 4-BIT MODEL WITH FLATQUANT QUANTIZERS")
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
    
    # THIS IS THE KEY: Get the quantizers object
    print("Applying FlatQuant W4 quantization and extracting quantizers...")
    quantizers = gptq_utils.rtn_fwrd(model, device, flatquant_args)
    model.eval()
    
    print(f"✅ Got quantizers for {len(quantizers)} layers")
    
    # Analyze quantizers to understand the structure
    print("\n🔍 Analyzing quantizers structure:")
    for name, quantizer in list(quantizers.items())[:3]:  # Show first 3
        print(f"  {name}:")
        print(f"    Type: {type(quantizer)}")
        if hasattr(quantizer, 'scale'):
            print(f"    Scale shape: {quantizer.scale.shape}")
        if hasattr(quantizer, 'zero'):
            print(f"    Zero shape: {quantizer.zero.shape}")
        if hasattr(quantizer, 'maxq'):
            print(f"    Max quantized value: {quantizer.maxq}")
    
    # Now use the quantizers to properly convert weights to 4-bit indices
    compressed_data = {}
    total_original_size = 0
    total_compressed_size = 0
    
    print(f"\n📦 Converting weights to 4-bit indices using quantizers:")
    
    for name, param in model.named_parameters():
        param_data = param.data.cpu()
        layer_name = None
        
        # Find corresponding quantizer
        for q_name in quantizers.keys():
            if q_name in name:  # Match quantizer to parameter
                layer_name = q_name
                break
        
        if layer_name and layer_name in quantizers:
            quantizer = quantizers[layer_name]
            print(f"  Processing {name} with quantizer {layer_name}")
            
            # Use quantizer to convert weights to 4-bit indices
            # This is the correct way to get the actual quantized indices
            with torch.no_grad():
                # Get the original weight from the quantizer if available
                # Extract quantization parameters
                scale = quantizer.scale
                zero = quantizer.zero if hasattr(quantizer, 'zero') else torch.zeros_like(scale)
                maxq = quantizer.maxq
                
                # Manual quantization using FlatQuant's exact method
                # Following flatquant.quant_utils.sym_quant()
                param_gpu = param_data.to(device)
                scale_gpu = scale.to(param_gpu.device)  # Ensure same device as param
                
                # FlatQuant uses symmetric quantization: q = clamp(round(x / scale), -(maxq+1), maxq)
                from flatquant.quant_utils import round_ste
                w_div_scale = param_gpu / scale_gpu
                
                # Create scalar tensors on the same device for clamp bounds
                min_bound = torch.tensor(-(maxq + 1), device=param_gpu.device, dtype=param_gpu.dtype)
                max_bound = torch.tensor(maxq, device=param_gpu.device, dtype=param_gpu.dtype)
                w_quantized = torch.clamp(round_ste(w_div_scale), min_bound, max_bound)
                
                # Store the 4-bit indices and quantization parameters  
                compressed_data[f"{name}_indices"] = w_quantized.cpu().to(torch.int8)  # Store as int8 for now
                compressed_data[f"{name}_scale"] = scale.cpu()
                # Note: FlatQuant symmetric quantization doesn't use zero offset
                compressed_data[f"{name}_shape"] = torch.tensor(param_data.shape, dtype=torch.int64)
                compressed_data[f"{name}_maxq"] = torch.tensor([maxq], dtype=torch.int32)
                
                # Calculate sizes
                original_size = param_data.numel() * 2  # BF16
                # 4-bit indices: 0.5 bytes per element + metadata (can pack later)
                index_size = w_quantized.numel() * 1  # int8 for now, can optimize to 0.5 bytes
                metadata_size = scale.numel() * 4 + zero.numel() * 4 + 4  # scale + zero + maxq
                compressed_size = index_size + metadata_size
                
                total_original_size += original_size
                total_compressed_size += compressed_size
                
                compression = original_size / compressed_size
                print(f"    Compressed: {original_size} -> {compressed_size} bytes ({compression:.1f}x)")
        else:
            # Keep non-quantized parameters (like normalization layers) in original format
            print(f"  Keeping {name} in BF16 (no quantizer found)")
            compressed_data[name] = param_data
            size = param_data.numel() * 2
            total_original_size += size
            total_compressed_size += size
    
    # Calculate overall compression
    overall_compression = total_original_size / total_compressed_size
    compressed_size_gb = total_compressed_size / (1024**3)
    
    print(f"\n📊 Compression Results:")
    print(f"Original size: {total_original_size/(1024**3):.2f} GB")
    print(f"Compressed size: {compressed_size_gb:.2f} GB")
    print(f"Overall compression: {overall_compression:.2f}x")
    
    # Save the properly compressed model
    output_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-proper-4bit/model_proper_4bit.safetensors"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\n💾 Saving to: {output_path}")
    save_file(compressed_data, output_path)
    
    actual_size = os.path.getsize(output_path)
    actual_size_gb = actual_size / (1024**3)
    print(f"Actual file size: {actual_size_gb:.2f} GB")
    
    target_size = 2.30 / 4  # 0.575 GB for 4:1 compression
    ratio = actual_size_gb / target_size
    print(f"Target size: {target_size:.2f} GB")
    print(f"Size ratio: {ratio:.2f}x (1.0x = perfect)")
    
    if ratio <= 1.1:
        print("✅ Compression target achieved!")
    else:
        print("⚠️  Compression target not quite reached")
    
    # Save quantizers for reconstruction
    quantizer_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-proper-4bit/quantizers.pth"
    torch.save(quantizers, quantizer_path)
    print(f"💾 Saved quantizers to: {quantizer_path}")
    
    return output_path, quantizer_path, actual_size_gb

if __name__ == "__main__":
    model_path, quantizer_path, file_size = create_proper_4bit_model()
    
    print(f"\n" + "="*70)
    print("✅ PROPER 4-BIT MODEL CREATED")
    print("="*70)
    print(f"📁 Model: {model_path}")
    print(f"📁 Quantizers: {quantizer_path}")
    print(f"📏 Size: {file_size:.2f} GB")
    print(f"🎯 Uses FlatQuant's actual quantization parameters")
    print(f"📝 Ready for testing with proper reconstruction")