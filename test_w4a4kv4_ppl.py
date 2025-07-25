#!/usr/bin/env python3

import torch
import sys
import os
import time

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')
sys.path.append('/workspace/FlatQuant/flatquant')

import flatquant.model_utils as model_utils
import flatquant.utils as utils
import gptq_utils
from transformers import AutoTokenizer

def evaluate_perplexity_simple(model, tokenizer, device, dataset_size=100):
    """Simple perplexity evaluation using test sentences"""
    test_texts = [
        "The capital of France is Paris, which is located in the northern central part of the country.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "The quick brown fox jumps over the lazy dog in the meadow during sunset.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "The human brain contains approximately 86 billion neurons that process information.",
        "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
        "Shakespeare wrote many famous plays including Romeo and Juliet and Hamlet.",
        "The Internet has revolutionized communication, commerce, and access to information worldwide.",
        "Quantum computing represents a fundamental shift in how we process complex calculations.",
    ] * (dataset_size // 10)
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    print(f"Evaluating perplexity on {len(test_texts)} text samples...")
    
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            if i % 20 == 0:
                print(f"  Processing sample {i+1}/{len(test_texts)}...")
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            
            # Accumulate loss and token count
            total_loss += loss.item() * inputs.input_ids.size(1)
            total_tokens += inputs.input_ids.size(1)
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity, avg_loss

def test_w4a4kv4_ppl():
    """Test the actual W4A4KV4 perplexity from FlatQuant"""
    print("🧪 TESTING ACTUAL W4A4KV4 PERPLEXITY")
    print("="*60)
    
    device = 'cuda'
    model_path = "./modelzoo/llama-3.2-1b"
    
    # Load model and apply FlatQuant W4A4KV4
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
            self.a_bits = 4  # 4-bit activations
            self.k_bits = 4  # 4-bit K
            self.k_asym = True
            self.k_groupsize = 128
            self.v_bits = 4  # 4-bit V  
            self.v_asym = True
            self.v_groupsize = 128
            self.gptq = False
            self.gptq_mse = False
    
    flatquant_args = FlatQuantArgs()
    model = model.to(device)
    
    # Apply W4A4KV4 quantization with transformation matrices
    matrix_path = "./outputs/llama-3.2-1b/w4a4/exp/flat_matrices.pth"
    
    if os.path.exists(matrix_path):
        print("Loading transformation matrices...")
        ckpt = torch.load(matrix_path, map_location='cpu')
        utils.add_dict(model, ckpt)
        print("✅ Loaded transformation matrices")
    else:
        print("⚠️  No transformation matrices found, using RTN only")
    
    # Apply quantization
    quantizers = gptq_utils.rtn_fwrd(model, device, flatquant_args)
    model.eval()
    
    print("✅ Applied FlatQuant W4A4KV4 quantization")
    
    # Evaluate perplexity
    start_time = time.time()
    w4a4kv4_ppl, w4a4kv4_loss = evaluate_perplexity_simple(model, tokenizer, device)
    eval_time = time.time() - start_time
    
    print(f"\n📊 W4A4KV4 Results:")
    print(f"   Perplexity: {w4a4kv4_ppl:.2f}")
    print(f"   Loss: {w4a4kv4_loss:.4f}")
    print(f"   Evaluation time: {eval_time:.1f}s")
    
    # Quick generation test
    print(f"\n🧪 Generation Test:")
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated = model.generate(
            inputs.input_ids,
            max_new_tokens=15,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Generation: {generated_text}")
    
    return w4a4kv4_ppl, w4a4kv4_loss

if __name__ == "__main__":
    ppl, loss = test_w4a4kv4_ppl()
    print(f"\n✅ ACTUAL W4A4KV4 PERPLEXITY: {ppl:.2f}")