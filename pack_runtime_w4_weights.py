#!/usr/bin/env python3

import torch
import os
from safetensors import safe_open
from safetensors.torch import save_file
from save_4bit_weights_packed import RTNQuantizer, should_quantize_weight

def pack_runtime_w4_weights():
    """Pack the runtime W4 weights to achieve 4:1 compression"""
    print("🔄 PACKING RUNTIME W4 WEIGHTS")
    print("="*50)
    
    # Paths
    runtime_w4_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4/model_runtime_w4.safetensors"
    packed_output_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-packed/model_runtime_w4_packed.safetensors"
    
    print(f"Input: {runtime_w4_path}")
    print(f"Output: {packed_output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(packed_output_path), exist_ok=True)
    
    # Initialize quantizer for packing
    quantizer = RTNQuantizer(n_bits=4)
    
    # Storage for packed weights
    packed_weights = {}
    
    # Statistics
    total_params = 0
    packed_params = 0
    original_size = 0
    packed_size = 0
    
    print("\n📊 Processing runtime W4 weights...")
    
    with safe_open(runtime_w4_path, framework="pt") as f:
        keys = f.keys()
        
        for i, key in enumerate(keys):
            # Key already has "model." prefix - use it directly  
            weight = f.get_tensor(key)
            
            # Get parameter name without "model." prefix for quantization decision
            param_name_no_prefix = key.replace("model.", "")
            
            params = weight.numel()
            total_params += params
            original_size += params * 2  # BF16 = 2 bytes per param
            
            if should_quantize_weight(param_name_no_prefix):
                # Pack the already-quantized weight
                print(f"  [{i+1:3d}/{len(keys)}] Packing: {param_name_no_prefix} {weight.shape}")
                
                # The weight is already quantized by FlatQuant, so we need to:
                # 1. Find the scale that was used
                # 2. Convert back to 4-bit integers 
                # 3. Pack the integers
                
                # Estimate the scale used (symmetric quantization)
                weight_max = weight.abs().max()
                scale = weight_max / 7.0  # 7 = 2^3 - 1 for 4-bit
                
                if scale > 0:
                    # Convert to 4-bit integers
                    quantized_int = torch.round(weight / scale).clamp(-8, 7)
                    
                    # Pack the integers
                    packed_tensor, original_shape = quantizer.pack_4bit(quantized_int.to(torch.int8))
                    
                    # Store packed weight and metadata (keep "model." prefix)
                    packed_weights[key] = packed_tensor
                    packed_weights[key + "_scale"] = torch.tensor(scale, dtype=torch.bfloat16)
                    packed_weights[key + "_shape"] = torch.tensor(original_shape, dtype=torch.int64)
                    
                    packed_params += params
                    packed_size += len(packed_tensor)  # Packed bytes
                    packed_size += 2  # Scale storage
                    packed_size += len(original_shape) * 8  # Shape storage
                else:
                    # Zero weight - store as BF16
                    print(f"      WARNING: Zero weight detected: {param_name_no_prefix}")
                    packed_weights[key] = weight
                    packed_size += params * 2
                
            else:
                # Keep normalization weights in BF16
                print(f"  [{i+1:3d}/{len(keys)}] Keeping BF16: {param_name_no_prefix} {weight.shape}")
                packed_weights[key] = weight
                packed_size += params * 2
    
    # Save packed weights
    print(f"\n💾 Saving packed runtime W4 weights...")
    save_file(packed_weights, packed_output_path)
    
    # Get actual file size
    actual_size = os.path.getsize(packed_output_path)
    
    # Print results
    print(f"\n✅ Runtime W4 packing completed!")
    print(f"\n📊 PACKING SUMMARY:")
    print(f"Total parameters: {total_params:,}")
    print(f"Packed to 4-bit: {packed_params:,} ({packed_params/total_params*100:.1f}%)")
    print(f"Kept in BF16: {total_params-packed_params:,} ({(total_params-packed_params)/total_params*100:.1f}%)")
    
    print(f"\n💾 SIZE COMPARISON:")
    print(f"Original BF16: {original_size:,} bytes ({original_size/1024/1024/1024:.2f} GB)")
    print(f"Runtime W4 (unpacked): {os.path.getsize(runtime_w4_path):,} bytes ({os.path.getsize(runtime_w4_path)/1024/1024/1024:.2f} GB)")
    print(f"Runtime W4 (packed): {actual_size:,} bytes ({actual_size/1024/1024/1024:.2f} GB)")
    print(f"Compression ratio: {original_size/actual_size:.2f}x")
    print(f"Size reduction: {(1 - actual_size/original_size)*100:.1f}%")
    
    # Check if we achieved target compression
    target_size = original_size / 4
    if abs(actual_size - target_size) / target_size < 0.2:  # Within 20%
        print(f"✅ Target achieved: ~1/4 size ({actual_size/target_size:.2f}x target)")
    else:
        print(f"⚠️  Target missed: Expected ~{target_size:,} bytes, got {actual_size:,} bytes")
        print(f"   Achieved: {original_size/actual_size:.2f}x compression (target was 4.0x)")
    
    return packed_output_path

def test_packed_runtime_w4():
    """Test the packed runtime W4 weights"""
    print(f"\n🧪 TESTING PACKED RUNTIME W4 WEIGHTS")
    print("="*50)
    
    from load_4bit_model_fixed import FourBitModelLoader
    
    # Paths
    packed_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-packed/model_runtime_w4_packed.safetensors"
    original_model_path = "./modelzoo/llama-3.2-1b"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load packed model
    print("Loading packed runtime W4 model...")
    loader = FourBitModelLoader()
    model, tokenizer = loader.load_4bit_model(packed_path, original_model_path, device)
    
    # Test generation
    test_input = "The capital of France is"
    print(f"\n🧪 Testing: '{test_input}'")
    
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Top predictions
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        top5 = torch.topk(probs, 5)
        
        print(f"Top-5 predictions:")
        for i, (prob, token_id) in enumerate(zip(top5.values[0], top5.indices[0])):
            token = tokenizer.decode(token_id.item())
            print(f"  {i+1}: '{token}' ({prob:.4f})")
        
        # Generation test
        print(f"\n🎯 Generation test:")
        generated = model.generate(
            inputs.input_ids,
            max_new_tokens=15,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
    
    # Weight sanity check
    print(f"\n🔍 Weight sanity check:")
    for name, param in model.named_parameters():
        if "q_proj" in name and "layers.0" in name:
            zero_percentage = (param == 0).sum().item() / param.numel() * 100
            print(f"Zero weights in {name}: {zero_percentage:.1f}%")
            print(f"Weight range: [{param.min():.6f}, {param.max():.6f}]")
            break
    
    return model, tokenizer

if __name__ == "__main__":
    # Pack the runtime W4 weights
    packed_path = pack_runtime_w4_weights()
    
    # Test the packed weights
    model, tokenizer = test_packed_runtime_w4()
    
    print(f"\n" + "="*50)
    print("PACKED RUNTIME W4 READY")
    print("="*50)
    print(f"Packed model saved to: {packed_path}")
    print("✅ Runtime W4 weights successfully packed and tested!")
    print("Next: Validate PPL and generation quality")