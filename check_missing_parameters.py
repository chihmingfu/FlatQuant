#!/usr/bin/env python3

import torch
import os
from safetensors import safe_open
import json

def check_missing_parameters():
    """Check what's missing from our compressed model"""
    print("🔍 CHECKING MISSING FLATQUANT PARAMETERS")
    print("="*50)
    
    # Load flat_parameters.pth
    flat_params_path = "/workspace/FlatQuant/outputs/llama-3.2-1b/w4a4/exp/flat_parameters.pth"
    flat_matrices_path = "/workspace/FlatQuant/outputs/llama-3.2-1b/w4a4/exp/flat_matrices.pth"
    
    print("📂 Loading original FlatQuant files...")
    
    if os.path.exists(flat_params_path):
        try:
            flat_params = torch.load(flat_params_path, map_location='cpu', weights_only=False)
            flat_params_size = os.path.getsize(flat_params_path) / (1024**2)
            print(f"✅ flat_parameters.pth: {flat_params_size:.1f} MB")
            print(f"   Keys: {list(flat_params.keys()) if isinstance(flat_params, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"❌ Error loading flat_parameters.pth: {e}")
            flat_params = None
    else:
        print(f"❌ flat_parameters.pth not found")
        flat_params = None
    
    if os.path.exists(flat_matrices_path):
        try:
            flat_matrices = torch.load(flat_matrices_path, map_location='cpu', weights_only=False)
            flat_matrices_size = os.path.getsize(flat_matrices_path) / (1024**2)
            print(f"✅ flat_matrices.pth: {flat_matrices_size:.1f} MB")
            print(f"   Keys: {len(flat_matrices.keys()) if isinstance(flat_matrices, dict) else 'Not a dict'}")
        except Exception as e:
            print(f"❌ Error loading flat_matrices.pth: {e}")
            flat_matrices = None
    else:
        print(f"❌ flat_matrices.pth not found")
        flat_matrices = None
    
    # Check what's in our compressed model
    print(f"\n📦 Checking our compressed model...")
    compressed_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit-compressed/model_true_4bit_compressed.safetensors"
    
    compressed_matrices_count = 0
    has_parameters = False
    
    with safe_open(compressed_path, framework="pt") as f:
        for key in f.keys():
            if key.startswith('flatquant_matrix_'):
                compressed_matrices_count += 1
            elif 'parameter' in key.lower():
                has_parameters = True
        
        # Check metadata
        safetensors_metadata = f.metadata()
        if safetensors_metadata:
            for meta_key in safetensors_metadata.keys():
                if 'parameter' in meta_key.lower():
                    has_parameters = True
    
    print(f"   FlatQuant matrices in compressed model: {compressed_matrices_count}")
    print(f"   FlatQuant parameters in compressed model: {'✅ Found' if has_parameters else '❌ Missing'}")
    
    # Analysis
    print(f"\n📊 ANALYSIS:")
    if flat_params and not has_parameters:
        print(f"❌ CRITICAL: flat_parameters.pth ({flat_params_size:.1f} MB) is MISSING from compressed model!")
        print(f"   This contains:")
        
        if isinstance(flat_params, dict):
            for key, value in flat_params.items():
                if hasattr(value, 'shape'):
                    size_mb = value.numel() * value.element_size() / (1024**2)
                    print(f"   - {key}: {value.shape} ({size_mb:.2f} MB)")
                else:
                    print(f"   - {key}: {type(value)}")
        
        print(f"\n🔧 IMPACT:")
        print(f"   Without flat_parameters.pth, the model may not load correctly")
        print(f"   It contains training state and learned parameters")
        print(f"   Current compressed model may be incomplete!")
        
        return False
    elif compressed_matrices_count == 0:
        print(f"❌ CRITICAL: No FlatQuant matrices found in compressed model!")
        return False
    else:
        print(f"✅ All required FlatQuant data appears to be included")
        return True

def check_actual_model_completeness():
    """Test if our current compressed model actually works properly"""
    print(f"\n🧪 TESTING CURRENT COMPRESSED MODEL COMPLETENESS...")
    
    # Try to load the compressed model and see if it needs flat_parameters
    sys.path.append('/workspace/FlatQuant')
    from load_compressed_model_fixed import load_compressed_model_fixed
    
    try:
        device = 'cuda'
        compressed_model_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit-compressed/model_true_4bit_compressed.safetensors"
        model, tokenizer = load_compressed_model_fixed(compressed_model_path, device)
        
        # Quick test
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs.input_ids)
        
        print(f"✅ Current compressed model loads and runs successfully")
        print(f"   Output shape: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Current compressed model has issues: {e}")
        return False

if __name__ == "__main__":
    missing_data = not check_missing_parameters()
    model_works = check_actual_model_completeness()
    
    if missing_data and model_works:
        print(f"\n🤔 CONCLUSION: Model works despite missing flat_parameters.pth")
        print(f"    This suggests flat_parameters.pth may not be needed for inference")
    elif missing_data and not model_works:
        print(f"\n❌ CONCLUSION: Missing flat_parameters.pth is causing issues")
    else:
        print(f"\n✅ CONCLUSION: Compressed model appears complete")