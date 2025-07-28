#!/usr/bin/env python3

import torch
import os

def examine_flat_parameters():
    """Examine what's inside flat_parameters.pth"""
    print("🔍 EXAMINING FLATQUANT PARAMETERS")
    print("="*50)
    
    flat_params_path = "/workspace/FlatQuant/outputs/llama-3.2-1b/w4a4/exp/flat_parameters.pth"
    flat_matrices_path = "/workspace/FlatQuant/outputs/llama-3.2-1b/w4a4/exp/flat_matrices.pth"
    
    if not os.path.exists(flat_params_path):
        print(f"❌ flat_parameters.pth not found")
        return
    
    print("📂 Loading flat_parameters.pth...")
    try:
        flat_params = torch.load(flat_params_path, map_location='cpu', weights_only=False)
        file_size = os.path.getsize(flat_params_path) / (1024**2)
        print(f"✅ Loaded flat_parameters.pth ({file_size:.1f} MB)")
        
        print(f"\n📊 STRUCTURE:")
        print(f"   Type: {type(flat_params)}")
        
        if isinstance(flat_params, dict):
            print(f"   Keys: {list(flat_params.keys())}")
            
            total_size_mb = 0
            
            # Examine each layer's parameters
            for layer_idx, layer_data in flat_params.items():
                print(f"\n🔍 Layer {layer_idx}:")
                print(f"   Type: {type(layer_data)}")
                
                if isinstance(layer_data, dict):
                    layer_size_mb = 0
                    print(f"   Keys: {list(layer_data.keys())}")
                    
                    for param_name, param_value in layer_data.items():
                        if hasattr(param_value, 'shape'):
                            param_size = param_value.numel() * param_value.element_size() / (1024**2)
                            layer_size_mb += param_size
                            total_size_mb += param_size
                            
                            print(f"   - {param_name}:")
                            print(f"     Shape: {param_value.shape}")
                            print(f"     Dtype: {param_value.dtype}")
                            print(f"     Size: {param_size:.3f} MB")
                            
                            # Show some sample values
                            if param_value.numel() <= 10:
                                print(f"     Values: {param_value.flatten()[:5].tolist()}")
                            else:
                                print(f"     Sample: {param_value.flatten()[:3].tolist()}...")
                        else:
                            print(f"   - {param_name}: {type(param_value)} = {param_value}")
                    
                    print(f"   Layer {layer_idx} total: {layer_size_mb:.3f} MB")
                    
                    # Only show first few layers to avoid spam
                    if layer_idx >= 2:
                        remaining_layers = len(flat_params) - layer_idx - 1
                        if remaining_layers > 0:
                            print(f"   ... ({remaining_layers} more layers)")
                        break
            
            print(f"\n📊 TOTAL PARAMETER DATA: {total_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error loading flat_parameters.pth: {e}")
        return
    
    # Compare with flat_matrices.pth
    print(f"\n🔍 COMPARING WITH FLAT_MATRICES.PTH:")
    
    if os.path.exists(flat_matrices_path):
        try:
            flat_matrices = torch.load(flat_matrices_path, map_location='cpu', weights_only=False)
            matrices_size = os.path.getsize(flat_matrices_path) / (1024**2)
            print(f"✅ flat_matrices.pth ({matrices_size:.1f} MB)")
            
            if isinstance(flat_matrices, dict):
                print(f"   Matrix count: {len(flat_matrices)}")
                matrices_total_mb = 0
                
                for matrix_name, matrix_tensor in flat_matrices.items():
                    if hasattr(matrix_tensor, 'shape'):
                        matrix_size = matrix_tensor.numel() * matrix_tensor.element_size() / (1024**2)
                        matrices_total_mb += matrix_size
                        print(f"   - {matrix_name}: {matrix_tensor.shape} ({matrix_size:.3f} MB)")
                
                print(f"   Total matrix data: {matrices_total_mb:.1f} MB")
        
        except Exception as e:
            print(f"❌ Error loading flat_matrices.pth: {e}")
    
    # Explanation
    print(f"\n💡 WHAT ARE FLATQUANT PARAMETERS?")
    print(f"   Based on the structure, flat_parameters.pth appears to contain:")
    print(f"   📊 Per-layer learned parameters from FlatQuant training")
    print(f"   🎯 These are likely the results of the calibration process")
    print(f"   🔧 Could include: optimization states, learned scales, biases")
    print(f"   📈 Training checkpoints or intermediate optimization results")
    
    print(f"\n🤔 DO WE NEED THEM FOR INFERENCE?")
    print(f"   Our compressed model works without them, suggesting:")
    print(f"   ✅ They may be training artifacts (optimizer states, etc.)")
    print(f"   ✅ The important data is in flat_matrices.pth (transformation matrices)")
    print(f"   ⚠️  OR they contain critical inference parameters we're missing")

if __name__ == "__main__":
    examine_flat_parameters()