#!/usr/bin/env python3

import torch
import os
import sys
from safetensors.torch import save_file
from collections import OrderedDict

def pack_4bit_indices(indices_tensor):
    """
    Pack 4-bit indices into bytes (2 indices per byte)
    Input: tensor with values in range [-8, 7] (signed 4-bit)
    Output: packed bytes + metadata for unpacking
    """
    print(f"    Packing tensor shape: {indices_tensor.shape}")
    
    # Convert signed indices [-7, 7] to unsigned [0, 15] for packing
    # Add 8 to shift range from [-7,7] to [1,15], then subtract 1 to get [0,14]
    # Actually, let's use a safer mapping: [-7,7] -> [0,14], leaving 15 as padding
    unsigned_indices = (indices_tensor + 7).clamp(0, 14).to(torch.uint8)
    
    # Flatten the tensor for packing
    flat_indices = unsigned_indices.flatten()
    original_numel = flat_indices.numel()
    
    # Pad with 15 (our padding value) if odd number of elements
    if flat_indices.numel() % 2 == 1:
        padding = torch.tensor([15], dtype=torch.uint8, device=flat_indices.device)
        flat_indices = torch.cat([flat_indices, padding])
    
    # Reshape into pairs and pack: low_nibble | (high_nibble << 4)
    pairs = flat_indices.view(-1, 2)
    packed_bytes = pairs[:, 0] | (pairs[:, 1] << 4)
    
    compression_ratio = original_numel / packed_bytes.numel()
    print(f"    Original elements: {original_numel:,}")
    print(f"    Packed bytes: {packed_bytes.numel():,}")
    print(f"    Compression ratio: {compression_ratio:.1f}x")
    
    return {
        'packed_data': packed_bytes.cpu(),
        'original_shape': list(indices_tensor.shape),
        'original_numel': original_numel,
        'dtype_info': 'signed_4bit_range_minus7_to_plus7'
    }

def unpack_4bit_indices(packed_data):
    """
    Unpack 4-bit indices from bytes
    """
    packed_bytes = packed_data['packed_data']
    original_shape = packed_data['original_shape']
    original_numel = packed_data['original_numel']
    
    # Unpack bytes into pairs of 4-bit values
    low_nibbles = packed_bytes & 0x0F  # Extract lower 4 bits
    high_nibbles = (packed_bytes >> 4) & 0x0F  # Extract upper 4 bits
    
    # Interleave to restore original order
    unpacked = torch.stack([low_nibbles, high_nibbles], dim=1).flatten()
    
    # Remove padding and restore original length
    unpacked = unpacked[:original_numel]
    
    # Convert back to signed indices: [0,14] -> [-7,7]
    signed_indices = (unpacked.to(torch.int8) - 7).clamp(-7, 7)
    
    # Restore original shape
    restored = signed_indices.view(original_shape)
    
    return restored

def create_true_4bit_compressed_model():
    """Create TRUE 4-bit compressed model with 2 indices per byte"""
    print("📦 CREATING TRUE 4-BIT COMPRESSED MODEL")
    print("="*60)
    
    # Load extracted data
    extracted_data_path = '/workspace/FlatQuant/extracted_quantized_data.pth'
    if not os.path.exists(extracted_data_path):
        print(f"❌ Extracted data not found at {extracted_data_path}")
        print("Please run extract_quantized_indices.py first")
        return
    
    print("📥 Loading extracted quantized data...")
    extracted_data = torch.load(extracted_data_path, map_location='cpu', weights_only=False)
    
    # Prepare compressed model data
    compressed_model = OrderedDict()
    total_original_size = 0
    total_compressed_size = 0
    
    print(f"\n📝 Processing embedding layer...")
    # Store embedding uncompressed
    embedding_data = extracted_data['embedding']
    compressed_model['embedding_weight'] = embedding_data['weight']
    embedding_size = embedding_data['weight'].numel() * embedding_data['weight'].element_size()
    total_original_size += embedding_size
    total_compressed_size += embedding_size
    print(f"   Embedding: {embedding_size/(1024**2):.1f} MB (uncompressed)")
    
    print(f"\n🔧 Processing {len(extracted_data['quantized_layers'])} quantized layers...")
    
    # Pack all quantized weights
    for layer_name, layer_data in extracted_data['quantized_layers'].items():
        print(f"\n  📦 Packing {layer_name}:")
        
        # Pack the 4-bit indices
        indices = layer_data['indices']
        packed_data = pack_4bit_indices(indices)
        
        # Store packed indices
        compressed_model[f"{layer_name}_packed_indices"] = packed_data['packed_data']
        compressed_model[f"{layer_name}_pack_metadata"] = {
            'original_shape': packed_data['original_shape'],
            'original_numel': packed_data['original_numel'],
            'dtype_info': packed_data['dtype_info']
        }
        
        # Store quantization metadata (scale, zero, maxq)
        compressed_model[f"{layer_name}_scale"] = layer_data['scale']
        compressed_model[f"{layer_name}_zero"] = layer_data['zero']
        compressed_model[f"{layer_name}_maxq"] = torch.tensor(layer_data['maxq'], dtype=torch.int32)
        
        # Calculate size savings
        original_size = indices.numel() * 2  # int8 = 1 byte, but original was BF16 = 2 bytes
        compressed_size = packed_data['packed_data'].numel()
        scale_zero_size = layer_data['scale'].numel() * layer_data['scale'].element_size() + \
                         layer_data['zero'].numel() * layer_data['zero'].element_size()
        
        total_original_size += original_size
        total_compressed_size += compressed_size + scale_zero_size
        
        print(f"    Packed indices: {compressed_size:,} bytes")
        print(f"    Scale/zero: {scale_zero_size:,} bytes")
        print(f"    Total layer compressed: {(compressed_size + scale_zero_size)/(1024**2):.2f} MB")
    
    # Store FlatQuant matrices
    print(f"\n🔄 Processing FlatQuant transformation matrices...")
    flatquant_matrices = extracted_data['flatquant_matrices']
    matrix_size = 0
    for key, value in flatquant_matrices.items():
        compressed_model[f"flatquant_matrix_{key}"] = value
        if hasattr(value, 'numel'):
            matrix_size += value.numel() * value.element_size()
        elif isinstance(value, dict):
            for k2, v2 in value.items():
                if hasattr(v2, 'numel'):
                    matrix_size += v2.numel() * v2.element_size()
    
    total_compressed_size += matrix_size
    print(f"   FlatQuant matrices: {matrix_size/(1024**2):.2f} MB")
    
    # Store metadata
    compressed_model['model_metadata'] = {
        'model_path': extracted_data['metadata']['model_path'],
        'quantization_config': extracted_data['metadata']['quantization_config'],
        'compression_info': {
            'format': 'true_4bit_packed',
            'indices_per_byte': 2,
            'index_range': [-7, 7],
            'total_layers': len(extracted_data['quantized_layers']),
            'original_size_mb': total_original_size / (1024**2),
            'compressed_size_mb': total_compressed_size / (1024**2),
            'compression_ratio': total_original_size / total_compressed_size
        }
    }
    
    print(f"\n💾 Saving compressed model...")
    
    # Create output directory
    output_dir = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit-compressed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as safetensors for efficient loading
    output_path = os.path.join(output_dir, "model_true_4bit_compressed.safetensors")
    
    # Convert non-tensor data to tensors for safetensors compatibility
    safetensors_data = OrderedDict()
    metadata_dict = {}
    
    for key, value in compressed_model.items():
        if isinstance(value, torch.Tensor):
            safetensors_data[key] = value
        elif key.endswith('_pack_metadata') or key == 'model_metadata':
            # Store complex metadata as JSON string in safetensors metadata
            import json
            metadata_dict[key] = json.dumps(value)
        else:
            # Convert other data to tensors if possible
            try:
                if isinstance(value, (int, float)):
                    safetensors_data[key] = torch.tensor(value)
                elif isinstance(value, (list, tuple)):
                    safetensors_data[key] = torch.tensor(value)
                else:
                    metadata_dict[key] = str(value)
            except:
                metadata_dict[key] = str(value)
    
    save_file(safetensors_data, output_path, metadata=metadata_dict)
    
    # Also save a separate metadata file for easier access
    metadata_path = os.path.join(output_dir, "compression_metadata.pth")
    torch.save(compressed_model['model_metadata'], metadata_path)
    
    # Get final file size
    file_size = os.path.getsize(output_path)
    
    print(f"\n📊 COMPRESSION SUMMARY:")
    print(f"   Original model size: ~{total_original_size/(1024**3):.2f} GB")
    print(f"   Compressed size: {total_compressed_size/(1024**2):.1f} MB")
    print(f"   File size on disk: {file_size/(1024**2):.1f} MB")
    print(f"   Compression ratio: {total_original_size/total_compressed_size:.1f}x")
    print(f"   Space saved: {(total_original_size-total_compressed_size)/(1024**2):.1f} MB")
    
    print(f"\n📂 FILES CREATED:")
    print(f"   Model: {output_path}")
    print(f"   Metadata: {metadata_path}")
    
    # Test unpacking one layer to verify correctness
    print(f"\n🧪 TESTING UNPACKING (first layer)...")
    first_layer = list(extracted_data['quantized_layers'].keys())[0]
    layer_data = extracted_data['quantized_layers'][first_layer]
    
    # Pack and unpack
    original_indices = layer_data['indices']
    packed_data = pack_4bit_indices(original_indices)
    unpacked_indices = unpack_4bit_indices(packed_data)
    
    # Verify
    if torch.equal(original_indices, unpacked_indices):
        print(f"   ✅ Pack/unpack test PASSED for {first_layer}")
    else:
        print(f"   ❌ Pack/unpack test FAILED for {first_layer}")
        print(f"      Original range: [{original_indices.min()}, {original_indices.max()}]")
        print(f"      Unpacked range: [{unpacked_indices.min()}, {unpacked_indices.max()}]")
        print(f"      Difference: {(original_indices - unpacked_indices).abs().sum()}")
    
    return output_path, total_compressed_size / (1024**2)

if __name__ == "__main__":
    output_path, compressed_size_mb = create_true_4bit_compressed_model()
    print(f"\n✅ Phase 3 completed!")
    print(f"📦 TRUE 4-bit compressed model: {compressed_size_mb:.1f} MB")