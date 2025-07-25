#!/usr/bin/env python3

import torch
import sys
import time
from transformers import AutoTokenizer, LlamaForCausalLM
from safetensors import safe_open

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

def main():
    print("🧪 EVALUATING DIRECT W4 MODEL PERPLEXITY")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test direct W4 model
    print(f"\n" + "="*60)
    print("DIRECT W4 MODEL PERPLEXITY TEST")
    print("="*60)
    
    # Load model structure
    original_model_path = "./modelzoo/llama-3.2-1b"
    direct_w4_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-direct/model_runtime_w4_direct.safetensors"
    
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    
    # Load direct W4 weights
    print("Loading direct W4 weights...")
    with safe_open(direct_w4_path, framework="pt") as f:
        keys = f.keys()
        replaced_count = 0
        
        for param_name, param in model.named_parameters():
            if param_name in keys:
                w4_weight = f.get_tensor(param_name)
                param.data.copy_(w4_weight)
                replaced_count += 1
    
    print(f"✅ Replaced {replaced_count} parameters")
    
    model = model.to(device)
    model.eval()
    
    # Evaluate perplexity
    start_time = time.time()
    direct_w4_ppl, direct_w4_loss = evaluate_perplexity_simple(model, tokenizer, device)
    eval_time = time.time() - start_time
    
    print(f"\n📊 Direct W4 Results:")
    print(f"   Perplexity: {direct_w4_ppl:.2f}")
    print(f"   Loss: {direct_w4_loss:.4f}")
    print(f"   Evaluation time: {eval_time:.1f}s")
    
    # Compare with expected W4A4KV4 result (~11.97)
    expected_ppl = 11.97
    difference = abs(direct_w4_ppl - expected_ppl)
    
    print(f"\n" + "="*60)
    print("COMPARISON WITH W4A4KV4 TARGET")
    print("="*60)
    print(f"Expected W4A4KV4 PPL: {expected_ppl:.2f}")
    print(f"Direct W4 PPL:       {direct_w4_ppl:.2f}")
    print(f"Difference:          {difference:.2f}")
    
    if difference < 1.0:
        print("✅ SUCCESS: PPL matches W4A4KV4 target (< 1.0 difference)")
        status = "PASS"
    elif difference < 3.0:
        print("⚠️  CLOSE: PPL is close to target (< 3.0 difference)")
        status = "CLOSE"
    else:
        print("❌ FAIL: PPL significantly different from target")
        status = "FAIL"
    
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
    
    print(f"\n✅ DIRECT W4 MODEL EVALUATION COMPLETED")
    print(f"📝 Status: {status}")
    
    return {
        'direct_w4_ppl': direct_w4_ppl,
        'direct_w4_loss': direct_w4_loss,
        'expected_ppl': expected_ppl,
        'difference': difference,
        'status': status
    }

if __name__ == "__main__":
    results = main()