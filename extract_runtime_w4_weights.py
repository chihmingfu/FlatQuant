#!/usr/bin/env python3

import torch
import sys
import os
from safetensors.torch import save_file
import time

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')
sys.path.append('/workspace/FlatQuant/flatquant')

import flatquant.model_utils as model_utils
import flatquant.utils as utils
import gptq_utils

def extract_runtime_w4_weights():
    """Extract weights from FlatQuant's actual runtime W4 quantization"""
    print("🔄 EXTRACTING RUNTIME W4 WEIGHTS FROM FLATQUANT")
    print("="*60)
    
    # Load model
    model_path = "./modelzoo/llama-3.2-1b"
    
    class Args:
        def __init__(self):
            self.model = model_path
            self.hf_token = None
    
    args = Args()
    print("Loading original model...")
    model, tokenizer = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    
    # Apply FlatQuant's W4 quantization (weights only)
    class FlatQuantArgs:
        def __init__(self):
            self.w_bits = 4
            self.w_asym = False
            self.w_groupsize = -1
            self.a_bits = 16  # Keep activations at 16-bit for this test
            self.k_bits = 16  # Keep KV at 16-bit for this test  
            self.v_bits = 16
            self.gptq = False
            self.gptq_mse = False  # Add missing attribute
    
    flatquant_args = FlatQuantArgs()
    
    print("Applying FlatQuant W4 quantization...")
    model = model.to(utils.DEV)  # Ensure model is on correct device
    quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, flatquant_args)
    
    print(f"✅ FlatQuant W4 quantization completed")
    model.eval()  # Ensure model is in eval mode
    
    # Extract the quantized weights
    print("Extracting quantized weights...")
    quantized_weights = {}
    
    for name, param in model.named_parameters():
        # Add "model." prefix to match original naming
        weight_key = f"model.{name}"
        quantized_weights[weight_key] = param.data.clone()
    
    print(f"Extracted {len(quantized_weights)} weights")
    
    # Calculate compression
    total_params = sum(w.numel() for w in quantized_weights.values())
    original_size = total_params * 2  # BF16 = 2 bytes per param
    
    # For 4-bit weights, we need to pack them to achieve true compression
    # But for now, let's save them as BF16 to verify behavior first
    print(f"\n📊 WEIGHT STATISTICS:")
    print(f"Total parameters: {total_params:,}")
    print(f"Memory usage (BF16): {original_size/1024/1024/1024:.2f} GB")
    
    # Save quantized weights
    output_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4/model_runtime_w4.safetensors"
    
    print(f"💾 Saving runtime W4 weights to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(quantized_weights, output_path)
    
    actual_size = os.path.getsize(output_path)
    print(f"Saved file size: {actual_size/1024/1024/1024:.2f} GB")
    
    return output_path, model, tokenizer

def test_runtime_w4_model():
    """Test the extracted runtime W4 model"""
    print(f"\n🧪 TESTING RUNTIME W4 MODEL")
    print("="*40)
    
    # Extract runtime W4 weights
    w4_model_path, quantized_model, original_tokenizer = extract_runtime_w4_weights()
    
    # Create a fresh tokenizer since the model may have been wrapped
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./modelzoo/llama-3.2-1b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test generation with the runtime quantized model
    test_input = "The capital of France is"
    print(f"Testing: '{test_input}'")
    
    inputs = tokenizer(test_input, return_tensors="pt").to(utils.DEV)
    
    with torch.no_grad():
        try:
            # Test forward pass
            outputs = quantized_model(**inputs)
            logits = outputs.logits
            print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            
            # Test generation (may not work with wrapped model)
            print("Attempting generation...")
            generated = quantized_model.generate(
                inputs.input_ids,
                max_new_tokens=15,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Generation failed (expected with wrapped model): {e}")
            # Just test logits
            outputs = quantized_model(**inputs)
            logits = outputs.logits
            print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            generated = None
    
    if generated is not None:
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
    else:
        print("Generation skipped due to model wrapper limitations")
    
    # Check top predictions
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    top5 = torch.topk(probs, 5)
    
    print(f"Top-5 predictions:")
    for i, (prob, token_id) in enumerate(zip(top5.values[0], top5.indices[0])):
        token = tokenizer.decode(token_id.item())
        print(f"  {i+1}: '{token}' ({prob:.4f})")
    
    return w4_model_path, quantized_model

def load_and_test_saved_w4():
    """Load the saved W4 weights and test them"""
    print(f"\n🔄 LOADING SAVED W4 WEIGHTS")
    print("="*40)
    
    from safetensors import safe_open
    from transformers import LlamaForCausalLM, AutoTokenizer
    
    # Paths
    w4_model_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4/model_runtime_w4.safetensors"
    original_model_path = "./modelzoo/llama-3.2-1b"
    
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
    
    # Load saved W4 weights
    print("Loading saved W4 weights...")
    with safe_open(w4_model_path, framework="pt") as f:
        keys = f.keys()
        
        replaced_count = 0
        for param_name, param in model.named_parameters():
            weight_key = f"model.{param_name}"
            if weight_key in keys:
                saved_weight = f.get_tensor(weight_key)
                param.data.copy_(saved_weight)
                replaced_count += 1
    
    print(f"Replaced {replaced_count} parameters")
    
    model = model.to(utils.DEV)
    model.eval()
    
    # Test the loaded model
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(utils.DEV)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        generated = model.generate(
            inputs.input_ids,
            max_new_tokens=15,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Loaded model generated: {generated_text}")
    
    return model, tokenizer

if __name__ == "__main__":
    print("🚀 RUNTIME W4 WEIGHT EXTRACTION TEST")
    print("="*60)
    
    # Step 1: Extract runtime W4 weights
    w4_path, runtime_model = test_runtime_w4_model()
    
    # Step 2: Load and test saved weights
    loaded_model, tokenizer = load_and_test_saved_w4()
    
    print(f"\n✅ Runtime W4 weight extraction completed!")
    print(f"Saved to: {w4_path}")
    print("This model should behave identically to runtime W4 quantization")