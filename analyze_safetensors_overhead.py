#!/usr/bin/env python3

import os
import json
from safetensors import safe_open

def analyze_safetensors_overhead():
    """Analyze the safetensors format overhead in our compressed model"""
    print("🔍 ANALYZING SAFETENSORS OVERHEAD")
    print("="*50)
    
    compressed_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit-compressed/model_true_4bit_compressed.safetensors"
    
    if not os.path.exists(compressed_path):
        print("❌ Compressed model not found")
        return
    
    total_file_size = os.path.getsize(compressed_path)
    print(f"📊 Total file size: {total_file_size / (1024**2):.1f} MB")
    
    # Analyze the contents
    tensor_sizes = {}
    metadata_size = 0
    total_tensor_data = 0
    
    with safe_open(compressed_path, framework="pt") as f:
        # Get all tensor keys and their sizes
        for key in f.keys():
            tensor = f.get_tensor(key)
            tensor_size_bytes = tensor.numel() * tensor.element_size()
            tensor_sizes[key] = tensor_size_bytes
            total_tensor_data += tensor_size_bytes
        
        # Get metadata
        safetensors_metadata = f.metadata()
        if safetensors_metadata:
            metadata_json = json.dumps(safetensors_metadata)
            metadata_size = len(metadata_json.encode('utf-8'))
    
    print(f"\n📊 CONTENT BREAKDOWN:")
    print(f"   Total tensor data: {total_tensor_data / (1024**2):.1f} MB")
    print(f"   Metadata size: {metadata_size / 1024:.1f} KB")
    
    # Calculate overhead
    safetensors_overhead = total_file_size - total_tensor_data - metadata_size
    overhead_mb = safetensors_overhead / (1024**2)
    overhead_percent = (safetensors_overhead / total_file_size) * 100
    
    print(f"   Safetensors overhead: {overhead_mb:.1f} MB ({overhead_percent:.1f}%)")
    
    print(f"\n🔍 WHAT IS SAFETENSORS OVERHEAD?")
    print(f"   Safetensors is a file format that stores:")
    print(f"   1. 📄 Header: File structure information")
    print(f"   2. 🗂️  Index: Tensor names, shapes, dtypes, offsets")
    print(f"   3. 🔗 Alignment: Memory alignment padding (usually 8-byte aligned)")
    print(f"   4. 📝 Metadata: JSON metadata we stored")
    print(f"   5. 💾 Tensor data: The actual tensor bytes")
    
    # Analyze by tensor category
    print(f"\n📊 TENSOR BREAKDOWN:")
    embedding_size = 0
    packed_indices_size = 0
    scales_zeros_size = 0
    matrices_size = 0
    other_size = 0
    
    for key, size in tensor_sizes.items():
        size_mb = size / (1024**2)
        if key == 'embedding_weight':
            embedding_size += size
            print(f"   📝 {key}: {size_mb:.1f} MB")
        elif 'packed_indices' in key:
            packed_indices_size += size
        elif 'scale' in key or 'zero' in key:
            scales_zeros_size += size
        elif 'flatquant_matrix' in key:
            matrices_size += size
        else:
            other_size += size
    
    print(f"   📦 Packed indices (all layers): {packed_indices_size / (1024**2):.1f} MB")
    print(f"   ⚖️  Scales & zeros (all layers): {scales_zeros_size / (1024**2):.1f} MB")
    print(f"   🔄 FlatQuant matrices: {matrices_size / (1024**2):.1f} MB")
    print(f"   📋 Other tensors: {other_size / (1024**2):.1f} MB")
    
    print(f"\n💡 WHY IS THERE OVERHEAD?")
    print(f"   • File format structure: Headers, indices, alignment")
    print(f"   • Each tensor has metadata: name, shape, dtype, offset")
    print(f"   • We have {len(tensor_sizes)} tensors, each needs index entry")
    print(f"   • Memory alignment padding between tensors")
    print(f"   • Our JSON metadata with compression info")
    
    print(f"\n🔧 COULD WE REDUCE OVERHEAD?")
    print(f"   ✅ We could combine multiple small tensors into arrays")
    print(f"   ✅ We could use custom binary format instead of safetensors")
    print(f"   ✅ We could compress the entire file with gzip/lz4")
    print(f"   ⚠️ But safetensors provides safety, mmap support, and compatibility")
    
    # Calculate theoretical minimum
    theoretical_minimum = embedding_size + packed_indices_size + scales_zeros_size + matrices_size
    theoretical_mb = theoretical_minimum / (1024**2)
    actual_overhead = total_file_size - theoretical_minimum
    actual_overhead_mb = actual_overhead / (1024**2)
    
    print(f"\n📊 THEORETICAL vs ACTUAL:")
    print(f"   Theoretical minimum (just tensor data): {theoretical_mb:.1f} MB")
    print(f"   Actual file size: {total_file_size / (1024**2):.1f} MB")
    print(f"   Total overhead: {actual_overhead_mb:.1f} MB")
    
    return overhead_mb, overhead_percent

if __name__ == "__main__":
    overhead_mb, overhead_percent = analyze_safetensors_overhead()
    print(f"\n✅ Analysis completed: {overhead_mb:.1f} MB overhead ({overhead_percent:.1f}%)")