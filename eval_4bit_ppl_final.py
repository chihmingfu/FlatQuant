#!/usr/bin/env python3

import torch
import sys
import time
from transformers import AutoTokenizer, LlamaForCausalLM
from load_4bit_model_fixed import FourBitModelLoader

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
    print("🧪 FINAL 4-BIT MODEL PERPLEXITY EVALUATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test original FP16 model first
    print(f"\n" + "="*60)
    print("1️⃣  ORIGINAL FP16 MODEL")
    print("="*60)
    
    original_model_path = "./modelzoo/llama-3.2-1b"
    
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    original_model = LlamaForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    original_model.eval()
    
    start_time = time.time()
    fp16_ppl, fp16_loss = evaluate_perplexity_simple(original_model, tokenizer, device)
    fp16_time = time.time() - start_time
    
    print(f"📊 FP16 Results:")
    print(f"   Perplexity: {fp16_ppl:.2f}")
    print(f"   Loss: {fp16_loss:.4f}")
    print(f"   Evaluation time: {fp16_time:.1f}s")
    
    # Test corrected 4-bit model
    print(f"\n" + "="*60)
    print("2️⃣  CORRECTED 4-BIT MODEL")
    print("="*60)
    
    fixed_4bit_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-packed/model_runtime_w4_fixed.safetensors"
    
    loader = FourBitModelLoader()
    model_4bit, _ = loader.load_4bit_model(fixed_4bit_path, original_model_path, device)
    
    start_time = time.time()
    model_4bit_ppl, model_4bit_loss = evaluate_perplexity_simple(model_4bit, tokenizer, device)
    model_4bit_time = time.time() - start_time
    
    print(f"📊 4-bit Results:")
    print(f"   Perplexity: {model_4bit_ppl:.2f}")
    print(f"   Loss: {model_4bit_loss:.4f}")
    print(f"   Evaluation time: {model_4bit_time:.1f}s")
    
    # Calculate degradation
    ppl_increase = (model_4bit_ppl - fp16_ppl) / fp16_ppl * 100
    
    print(f"\n" + "="*60)
    print("📊 FINAL COMPARISON")
    print("="*60)
    print(f"FP16 Perplexity:    {fp16_ppl:.2f}")
    print(f"4-bit Perplexity:   {model_4bit_ppl:.2f}")
    print(f"Perplexity increase: {ppl_increase:+.1f}%")
    print(f"Size reduction:     75.0% (4:1 compression)")
    
    # Weight verification
    print(f"\n🔍 4-bit Model Weight Verification:")
    for name, param in model_4bit.named_parameters():
        if "q_proj" in name and "layers.0" in name:
            zero_percentage = (param == 0).sum().item() / param.numel() * 100
            print(f"Zero weights in {name}: {zero_percentage:.1f}%")
            print(f"Weight range: [{param.min():.6f}, {param.max():.6f}]")
            print(f"Weight std: {param.std():.6f}")
            break
    
    print(f"\n✅ PERPLEXITY EVALUATION COMPLETED")
    print(f"📝 Results ready for final report update")
    
    return {
        'fp16_ppl': fp16_ppl,
        'fp16_loss': fp16_loss,
        '4bit_ppl': model_4bit_ppl,
        '4bit_loss': model_4bit_loss,
        'ppl_increase_percent': ppl_increase
    }

if __name__ == "__main__":
    results = main()