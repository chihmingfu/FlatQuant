#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from load_4bit_model_fixed import FourBitModelLoader

def test_fixed_4bit_model():
    """Test the fixed 4-bit model"""
    print("🧪 TESTING FIXED 4-BIT MODEL")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load fixed 4-bit model
    fixed_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-packed/model_runtime_w4_fixed.safetensors"
    original_model_path = "./modelzoo/llama-3.2-1b"
    
    print("Loading fixed 4-bit model...")
    loader = FourBitModelLoader()
    model, tokenizer = loader.load_4bit_model(fixed_path, original_model_path, device)
    
    # Test generation
    test_prompts = [
        "The capital of France is",
        "Machine learning is"
    ]
    
    print(f"\n🎯 Generation test:")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated = model.generate(
                inputs.input_ids,
                max_new_tokens=15,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"Prompt: '{prompt}'")
            print(f"Generated: {generated_text}")
            print()
    
    # Weight sanity check
    print(f"🔍 Weight sanity check:")
    for name, param in model.named_parameters():
        if "q_proj" in name and "layers.0" in name:
            zero_percentage = (param == 0).sum().item() / param.numel() * 100
            print(f"Zero weights in {name}: {zero_percentage:.1f}%")
            print(f"Weight range: [{param.min():.6f}, {param.max():.6f}]")
            print(f"Weight std: {param.std():.6f}")
            break
    
    return model, tokenizer

def compare_with_original():
    """Compare with original model to see if there's a difference"""
    print(f"\n🔍 COMPARING WITH ORIGINAL MODEL")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load original model
    print("Loading original FP16 model...")
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
    
    # Load fixed 4-bit model
    print("Loading fixed 4-bit model...")
    fixed_4bit_model, _ = test_fixed_4bit_model()
    
    # Test same input
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    print(f"\n📊 Comparing models on: '{test_input}'")
    
    with torch.no_grad():
        # Original model
        original_outputs = original_model(**inputs)
        original_logits = original_outputs.logits
        
        # Fixed 4-bit model
        fixed_outputs = fixed_4bit_model(**inputs)
        fixed_logits = fixed_outputs.logits
    
    # Compare logits
    logits_diff = torch.abs(original_logits - fixed_logits)
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    
    print(f"Max logits difference: {max_diff:.6f}")
    print(f"Mean logits difference: {mean_diff:.6f}")
    
    # Compare top predictions
    original_probs = torch.softmax(original_logits[:, -1, :], dim=-1)
    fixed_probs = torch.softmax(fixed_logits[:, -1, :], dim=-1)
    
    original_top5 = torch.topk(original_probs, 5)
    fixed_top5 = torch.topk(fixed_probs, 5)
    
    print(f"\nOriginal top-5:")
    for i, (prob, token_id) in enumerate(zip(original_top5.values[0], original_top5.indices[0])):
        token = tokenizer.decode(token_id.item())
        print(f"  {i+1}: '{token}' ({prob:.4f})")
    
    print(f"\nFixed 4-bit top-5:")
    for i, (prob, token_id) in enumerate(zip(fixed_top5.values[0], fixed_top5.indices[0])):
        token = tokenizer.decode(token_id.item())
        print(f"  {i+1}: '{token}' ({prob:.4f})")
    
    # Check if models are identical (indicating 4-bit loading failed)
    if max_diff < 1e-6:
        print(f"\n❌ MODELS ARE IDENTICAL!")
        print(f"   This suggests the 4-bit weights are not being loaded properly.")
        print(f"   The model is still using original FP16 weights.")
    else:
        print(f"\n✅ MODELS ARE DIFFERENT!")
        print(f"   This confirms the 4-bit weights are being used.")
        print(f"   Quantization effect detected: {max_diff:.6f} max difference")
    
    return max_diff > 1e-6

if __name__ == "__main__":
    model, tokenizer = test_fixed_4bit_model()
    using_4bit = compare_with_original()
    
    print(f"\n" + "="*50)
    print("FINAL STATUS")
    print("="*50)
    if using_4bit:
        print("✅ 4-bit model is working correctly!")
        print("✅ Quantized weights are being used!")
    else:
        print("❌ 4-bit model is NOT working - still using FP16 weights!")
        print("   Need to debug the weight loading mechanism further.")