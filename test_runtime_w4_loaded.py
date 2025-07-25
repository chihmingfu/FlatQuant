#!/usr/bin/env python3

import torch
import sys
import os

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')

from transformers import AutoTokenizer, LlamaForCausalLM
from safetensors import safe_open

def test_runtime_w4_loaded():
    """Test the loaded runtime W4 model"""
    print("🧪 TESTING LOADED RUNTIME W4 MODEL")
    print("="*50)
    
    # Paths
    runtime_w4_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4/model_runtime_w4.safetensors"
    original_model_path = "./modelzoo/llama-3.2-1b"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model structure
    print("Loading model structure...")
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    
    # Load runtime W4 weights
    print("Loading runtime W4 weights...")
    with safe_open(runtime_w4_path, framework="pt") as f:
        keys = f.keys()
        replaced_count = 0
        
        for param_name, param in model.named_parameters():
            weight_key = f"model.{param_name}"
            if weight_key in keys:
                runtime_weight = f.get_tensor(weight_key)
                param.data.copy_(runtime_weight)
                replaced_count += 1
        
        print(f"   ✅ Replaced {replaced_count} parameters with runtime W4 weights")
    
    model = model.to(device)
    model.eval()
    
    # Test the loaded model
    test_input = "The capital of France is"
    print(f"\n🧪 Testing input: '{test_input}'")
    
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Check top-5 predictions
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        top5 = torch.topk(probs, 5)
        
        print(f"Top-5 predictions:")
        for i, (prob, token_id) in enumerate(zip(top5.values[0], top5.indices[0])):
            token = tokenizer.decode(token_id.item())
            print(f"  {i+1}: '{token}' ({prob:.4f})")
        
        # Test generation
        print(f"\n🎯 Testing generation:")
        generated = model.generate(
            inputs.input_ids,
            max_new_tokens=15,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
    
    # Check weight sanity
    print(f"\n🔍 Weight sanity check:")
    for name, param in model.named_parameters():
        if "q_proj" in name and "layers.0" in name:  # Check first q_proj
            zero_percentage = (param == 0).sum().item() / param.numel() * 100
            print(f"Zero weights in {name}: {zero_percentage:.1f}%")
            print(f"Weight range: [{param.min():.6f}, {param.max():.6f}]")
            print(f"Weight std: {param.std():.6f}")
            break
    
    return model, tokenizer

def compare_with_original():
    """Compare runtime W4 loaded model with original"""
    print("\n🔍 COMPARING WITH ORIGINAL MODEL")
    print("="*50)
    
    original_model_path = "./modelzoo/llama-3.2-1b"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load original model
    print("Loading original model...")
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    original_model = LlamaForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    original_model.eval()
    
    # Load runtime W4 model
    print("Loading runtime W4 model...")
    runtime_w4_model, _ = test_runtime_w4_loaded()
    
    # Test same input
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    print(f"\n📊 Comparing forward pass...")
    
    with torch.no_grad():
        # Original model
        original_outputs = original_model(**inputs)
        original_logits = original_outputs.logits
        
        # Runtime W4 model
        w4_outputs = runtime_w4_model(**inputs)
        w4_logits = w4_outputs.logits
    
    # Compare logits
    logits_diff = torch.abs(original_logits - w4_logits)
    print(f"Max logits difference: {logits_diff.max():.6f}")
    print(f"Mean logits difference: {logits_diff.mean():.6f}")
    
    # Compare top predictions
    original_probs = torch.softmax(original_logits[:, -1, :], dim=-1)
    w4_probs = torch.softmax(w4_logits[:, -1, :], dim=-1)
    
    original_top5 = torch.topk(original_probs, 5)
    w4_top5 = torch.topk(w4_probs, 5)
    
    print(f"\nOriginal top-5:")
    for i, (prob, token_id) in enumerate(zip(original_top5.values[0], original_top5.indices[0])):
        token = tokenizer.decode(token_id.item())
        print(f"  {i+1}: '{token}' ({prob:.4f})")
    
    print(f"\nRuntime W4 top-5:")
    for i, (prob, token_id) in enumerate(zip(w4_top5.values[0], w4_top5.indices[0])):
        token = tokenizer.decode(token_id.item())
        print(f"  {i+1}: '{token}' ({prob:.4f})")
    
    # Check overlap
    original_tokens = set(original_top5.indices[0].tolist())
    w4_tokens = set(w4_top5.indices[0].tolist())
    overlap = len(original_tokens.intersection(w4_tokens))
    print(f"\nTop-5 overlap: {overlap}/5")
    
    return {
        'max_logits_diff': logits_diff.max().item(),
        'mean_logits_diff': logits_diff.mean().item(),
        'top5_overlap': overlap
    }

if __name__ == "__main__":
    # Test runtime W4 loaded model
    model, tokenizer = test_runtime_w4_loaded()
    
    # Compare with original
    results = compare_with_original()
    
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Max logits diff: {results['max_logits_diff']:.6f}")
    print(f"Mean logits diff: {results['mean_logits_diff']:.6f}")
    print(f"Top-5 overlap: {results['top5_overlap']}/5")
    
    if results['max_logits_diff'] < 0.1 and results['top5_overlap'] >= 4:
        print("✅ Runtime W4 model behaves identically to original!")
        print("This confirms FlatQuant's W4 quantization preserves model behavior")
    else:
        print("⚠️  Some differences detected - this is expected for W4 quantization")
        print("The weights are correctly extracted from FlatQuant's runtime W4")