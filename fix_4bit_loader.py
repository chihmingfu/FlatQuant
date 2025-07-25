#!/usr/bin/env python3

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os

def fix_packed_weights_naming():
    """Fix the parameter naming in packed weights"""
    print("🔧 FIXING PACKED WEIGHTS PARAMETER NAMING")
    print("="*50)
    
    # Paths
    current_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-packed/model_runtime_w4_packed.safetensors"
    fixed_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-packed/model_runtime_w4_fixed.safetensors"
    
    print(f"Input: {current_path}")
    print(f"Output: {fixed_path}")
    
    fixed_weights = {}
    
    with safe_open(current_path, framework="pt") as f:
        keys = f.keys()
        
        print(f"Processing {len(keys)} keys...")
        
        for key in keys:
            tensor = f.get_tensor(key)
            
            # Remove the extra "model." prefix if it exists
            if key.startswith("model.model."):
                # Fix: model.model.xxx -> model.xxx
                fixed_key = key.replace("model.model.", "model.")
                print(f"  Fixed: {key} -> {fixed_key}")
            else:
                # Keep as is
                fixed_key = key
            
            fixed_weights[fixed_key] = tensor
    
    # Save fixed weights
    print(f"\n💾 Saving fixed weights...")
    save_file(fixed_weights, fixed_path)
    
    actual_size = os.path.getsize(fixed_path)
    print(f"Fixed model saved: {actual_size/1024/1024/1024:.2f} GB")
    
    return fixed_path

if __name__ == "__main__":
    fixed_path = fix_packed_weights_naming()
    print(f"\n✅ Fixed model ready: {fixed_path}")