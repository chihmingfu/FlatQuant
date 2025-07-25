#!/usr/bin/env python3

import torch
import sys

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')
sys.path.append('/workspace/FlatQuant/flatquant')

import flatquant.model_utils as model_utils
import gptq_utils

def check_flatquant_quantized_layers():
    """Check which layers FlatQuant actually quantizes"""
    print("🔍 CHECKING FLATQUANT'S QUANTIZATION STRATEGY")
    print("="*60)
    
    device = 'cuda'
    model_path = "./modelzoo/llama-3.2-1b"
    
    # Load model
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
    
    print("Getting quantizers from FlatQuant...")
    quantizers = gptq_utils.rtn_fwrd(model, device, flatquant_args)
    
    print(f"\n📊 FlatQuant quantizes {len(quantizers)} layers:")
    quantized_layers = set(quantizers.keys())
    
    print("\n✅ QUANTIZED by FlatQuant:")
    for layer_name in sorted(quantized_layers):
        print(f"  {layer_name}")
    
    print(f"\n🔍 ALL model parameters:")
    all_params = []
    quantized_params = []
    not_quantized_params = []
    
    for name, param in model.named_parameters():
        all_params.append(name)
        
        # Check if this parameter has a corresponding quantizer
        has_quantizer = False
        for q_name in quantized_layers:
            if q_name in name:
                has_quantizer = True
                quantized_params.append(name)
                break
        
        if not has_quantizer:
            not_quantized_params.append(name)
    
    print(f"\n📈 SUMMARY:")
    print(f"Total parameters: {len(all_params)}")
    print(f"Quantized by FlatQuant: {len(quantized_params)}")
    print(f"NOT quantized by FlatQuant: {len(not_quantized_params)}")
    
    print(f"\n❌ NOT QUANTIZED by FlatQuant:")
    for param_name in not_quantized_params:
        param = dict(model.named_parameters())[param_name]
        param_size = param.numel() * 2 / (1024**2)  # Size in MB
        print(f"  {param_name} - {param.shape} - {param_size:.1f} MB")
    
    # Check specifically for embedding layer
    embed_quantized = any("embed_tokens" in q_name for q_name in quantized_layers)
    print(f"\n🎯 EMBEDDING LAYER ANALYSIS:")
    print(f"embed_tokens quantized by FlatQuant: {'✅ YES' if embed_quantized else '❌ NO'}")
    
    # Get embedding layer size
    embed_param = None
    for name, param in model.named_parameters():
        if "embed_tokens" in name:
            embed_param = param
            embed_size_mb = param.numel() * 2 / (1024**2)
            embed_size_gb = param.numel() * 2 / (1024**3)
            print(f"Embedding size: {embed_param.shape} - {embed_size_mb:.1f} MB ({embed_size_gb:.3f} GB)")
            break
    
    # Calculate impact of embedding on compression
    total_model_size = sum(p.numel() * 2 for p in model.parameters()) / (1024**3)
    if embed_param is not None:
        embed_percentage = (embed_param.numel() * 2 / (1024**3)) / total_model_size * 100
        print(f"Embedding is {embed_percentage:.1f}% of total model size")
    
    print(f"\n💡 CONCLUSION:")
    if embed_quantized:
        print("FlatQuant DOES quantize embedding layer - we should include it")
    else:
        print("FlatQuant does NOT quantize embedding layer - we should keep it in FP16")
        print("This explains why we can't reach exactly 4:1 compression")

if __name__ == "__main__":
    check_flatquant_quantized_layers()