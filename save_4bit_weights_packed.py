#!/usr/bin/env python3

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import os
import time

class RTNQuantizer:
    """Round-to-Nearest 4-bit quantizer with bit packing"""
    
    def __init__(self, n_bits=4):
        self.n_bits = n_bits
        self.max_val = 2 ** (n_bits - 1) - 1  # 7 for 4-bit
        self.min_val = -2 ** (n_bits - 1)     # -8 for 4-bit
    
    def quantize_tensor(self, tensor):
        """Quantize tensor to 4-bit using RTN and pack bits"""
        # Calculate scale and zero point
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        # Symmetric quantization
        scale = max(abs(tensor_min), abs(tensor_max)) / self.max_val
        
        # Quantize to 4-bit range
        quantized = torch.round(tensor / scale).clamp(self.min_val, self.max_val)
        
        # Convert to int8 and pack 2 values per byte
        quantized_int8 = quantized.to(torch.int8)
        packed = self.pack_4bit(quantized_int8)
        
        return packed, scale
    
    def pack_4bit(self, tensor_4bit):
        """Pack two 4-bit values into one uint8"""
        # Flatten the tensor for packing
        flat = tensor_4bit.flatten()
        
        # Add zero padding if odd length
        if len(flat) % 2 == 1:
            flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8)])
        
        # Reshape to pairs
        pairs = flat.view(-1, 2)
        
        # Pack: first value in lower 4 bits, second in upper 4 bits
        # Handle negative values by adding 16 to get unsigned representation
        val1 = (pairs[:, 0] + 16) % 16  # Convert -8..7 to 0..15
        val2 = (pairs[:, 1] + 16) % 16
        
        packed = val1 + (val2 << 4)
        
        # Store original shape for unpacking
        return packed.to(torch.uint8), tensor_4bit.shape
    
    def unpack_4bit(self, packed_tensor, original_shape):
        """Unpack uint8 values back to two 4-bit int8 values"""
        # Extract lower and upper 4 bits
        val1 = packed_tensor & 0x0F  # Lower 4 bits
        val2 = (packed_tensor >> 4) & 0x0F  # Upper 4 bits
        
        # Convert back to signed values (-8..7)
        val1 = (val1.to(torch.int8) - 8) % 16 - 8
        val2 = (val2.to(torch.int8) - 8) % 16 - 8
        
        # Interleave values
        unpacked = torch.stack([val1, val2], dim=1).flatten()
        
        # Trim to original size and reshape
        total_elements = torch.prod(torch.tensor(original_shape))
        unpacked = unpacked[:total_elements]
        
        return unpacked.view(original_shape)
    
    def dequantize_tensor(self, packed_tensor, original_shape, scale):
        """Unpack and dequantize tensor back to BF16"""
        unpacked = self.unpack_4bit(packed_tensor, original_shape)
        return unpacked.to(torch.bfloat16) * scale

def should_quantize_weight(weight_name):
    """Determine if a weight should be quantized to 4-bit"""
    # Quantize large linear weights
    quantize_patterns = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
        'embed_tokens'
    ]
    
    # Keep normalization weights in BF16
    keep_bf16_patterns = [
        'input_layernorm', 'post_attention_layernorm', 
        'norm.weight', 'layernorm', 'rms_norm'
    ]
    
    # Check if should keep in BF16
    if any(pattern in weight_name for pattern in keep_bf16_patterns):
        return False
    
    # Check if should quantize
    if any(pattern in weight_name for pattern in quantize_patterns):
        return True
    
    # Default: don't quantize unknown weights
    print(f"WARNING: Unknown weight type: {weight_name} - keeping in BF16")
    return False

def save_4bit_model_packed(input_path, output_path):
    """Save model with properly packed 4-bit quantized weights"""
    
    print("🔄 Starting 4-bit weight quantization with bit packing...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    quantizer = RTNQuantizer(n_bits=4)
    
    # Storage for quantized model
    quantized_weights = {}
    metadata = {}
    
    # Statistics
    total_params = 0
    quantized_params = 0
    original_size = 0
    packed_size = 0
    
    start_time = time.time()
    
    print("\n📊 Processing weights...")
    with safe_open(input_path, framework="pt") as f:
        keys = f.keys()
        
        for i, key in enumerate(keys):
            weight = f.get_tensor(key)
            params = weight.numel()
            total_params += params
            original_size += params * 2  # BF16 = 2 bytes per param
            
            if should_quantize_weight(key):
                # Quantize to 4-bit with bit packing
                print(f"  [{i+1:3d}/{len(keys)}] Quantizing: {key} {weight.shape}")
                packed_weight, scale = quantizer.quantize_tensor(weight)
                packed_tensor, original_shape = packed_weight
                
                # Store packed weights and metadata
                quantized_weights[key] = packed_tensor
                quantized_weights[key + "_scale"] = torch.tensor(scale, dtype=torch.bfloat16)
                
                # Store shape metadata as tensor for safetensors compatibility
                shape_tensor = torch.tensor(original_shape, dtype=torch.int64)
                quantized_weights[key + "_shape"] = shape_tensor
                
                quantized_params += params
                packed_size += len(packed_tensor)  # Packed size in bytes
                packed_size += 2  # Scale storage
                packed_size += len(original_shape) * 8  # Shape storage
                
            else:
                # Keep in BF16
                print(f"  [{i+1:3d}/{len(keys)}] Keeping BF16: {key} {weight.shape}")
                quantized_weights[key] = weight
                packed_size += params * 2  # BF16 = 2 bytes per param
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save quantized model
    print(f"\n💾 Saving packed quantized model to: {output_path}")
    save_file(quantized_weights, output_path)
    
    # Calculate actual file size
    actual_size = os.path.getsize(output_path)
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n✅ 4-bit quantization with packing completed!")
    print(f"⏱️  Time taken: {elapsed_time:.2f} seconds")
    print(f"\n📊 QUANTIZATION SUMMARY:")
    print(f"Total parameters: {total_params:,}")
    print(f"Quantized to 4-bit: {quantized_params:,} ({quantized_params/total_params*100:.1f}%)")
    print(f"Kept in BF16: {total_params-quantized_params:,} ({(total_params-quantized_params)/total_params*100:.1f}%)")
    
    print(f"\n💾 SIZE COMPARISON:")
    print(f"Original model: {original_size:,} bytes ({original_size/1024/1024/1024:.2f} GB)")
    print(f"Theoretical packed: {packed_size:,} bytes ({packed_size/1024/1024/1024:.2f} GB)")
    print(f"Actual saved file: {actual_size:,} bytes ({actual_size/1024/1024/1024:.2f} GB)")
    print(f"Compression ratio: {original_size/actual_size:.2f}x")
    print(f"Size reduction: {(1 - actual_size/original_size)*100:.1f}%")
    
    # Verify target (should be ~1/4 of original)
    target_size = original_size / 4
    if abs(actual_size - target_size) / target_size < 0.2:  # Within 20%
        print(f"✅ Target achieved: ~1/4 size ({actual_size/target_size:.2f}x target)")
    else:
        print(f"⚠️  Target missed: Expected ~{target_size:,} bytes, got {actual_size:,} bytes")
        print(f"   Compression: {original_size/actual_size:.2f}x (target was 4.0x)")
    
    return output_path

def test_packing():
    """Test the bit packing/unpacking functionality"""
    print("🧪 Testing bit packing...")
    quantizer = RTNQuantizer()
    
    # Create test tensor
    test_tensor = torch.randn(100, 200, dtype=torch.bfloat16)
    
    # Quantize
    packed, scale = quantizer.quantize_tensor(test_tensor)
    packed_tensor, original_shape = packed
    
    # Dequantize
    restored = quantizer.dequantize_tensor(packed_tensor, original_shape, scale)
    
    # Check compression
    original_bytes = test_tensor.numel() * 2
    packed_bytes = len(packed_tensor)
    compression = original_bytes / packed_bytes
    
    print(f"Original size: {original_bytes} bytes")
    print(f"Packed size: {packed_bytes} bytes")
    print(f"Compression: {compression:.2f}x")
    print(f"Shape preserved: {test_tensor.shape == restored.shape}")
    
    # Check quality
    mse = torch.mean((test_tensor - restored) ** 2)
    print(f"MSE: {mse:.6f}")
    print("✅ Packing test completed")

if __name__ == "__main__":
    # Test packing first
    test_packing()
    
    print("\n" + "="*60)
    
    # Paths
    input_model = "/workspace/FlatQuant/modelzoo/llama-3.2-1b/model.safetensors"
    output_dir = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-4bit-packed"
    output_model = os.path.join(output_dir, "model_4bit_packed.safetensors")
    
    # Create 4-bit model with proper packing
    saved_path = save_4bit_model_packed(input_model, output_model)
    
    print(f"\n🎯 READY FOR TESTING:")
    print(f"4-bit packed model saved to: {saved_path}")
    print("Next: Load this model and validate PPL performance")