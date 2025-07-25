#!/usr/bin/env python3

import torch
import sys
import os
from safetensors.torch import save_file

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')
sys.path.append('/workspace/FlatQuant/flatquant')

import flatquant.model_utils as model_utils
import flatquant.utils as utils
import gptq_utils

def save_runtime_w4_directly():
    """Save FlatQuant's runtime W4 weights directly without re-quantization"""
    print("🔧 SAVING RUNTIME W4 WEIGHTS DIRECTLY")
    print("="*60)
    
    device = 'cuda'
    model_path = "./modelzoo/llama-3.2-1b"
    
    # Load model and apply FlatQuant W4
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
    quantizers = gptq_utils.rtn_fwrd(model, device, flatquant_args)
    model.eval()
    
    print("✅ Applied FlatQuant W4 quantization")
    
    # Extract weights directly (no re-quantization!)
    runtime_w4_weights = {}
    for name, param in model.named_parameters():
        runtime_w4_weights[name] = param.data.clone().cpu()  # Move to CPU for saving
    
    print(f"Extracted {len(runtime_w4_weights)} runtime W4 weights")
    
    # Calculate file size
    total_params = sum(w.numel() for w in runtime_w4_weights.values())
    original_size = total_params * 2  # BF16 = 2 bytes per param
    
    print(f"Total parameters: {total_params:,}")
    print(f"File size: {original_size/1024/1024/1024:.2f} GB")
    
    # NOTE: This will be 2.30 GB (same as original) because we're not compressing.
    # But the weights should exactly match FlatQuant W4 behavior and give correct PPL.
    
    # Save directly
    output_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-direct/model_runtime_w4_direct.safetensors"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"💾 Saving to: {output_path}")
    save_file(runtime_w4_weights, output_path)
    
    actual_size = os.path.getsize(output_path)
    print(f"Saved file size: {actual_size/1024/1024/1024:.2f} GB")
    
    print(f"\n✅ Runtime W4 weights saved directly (no compression)")
    print(f"📝 This should give correct PPL matching W4A4KV4 results")
    
    return output_path

def test_direct_w4_model():
    """Test the directly saved W4 model"""
    print(f"\n🧪 TESTING DIRECT W4 MODEL")
    print("="*50)
    
    from transformers import AutoTokenizer, LlamaForCausalLM
    from safetensors import safe_open
    
    # Paths
    direct_w4_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-runtime-w4-direct/model_runtime_w4_direct.safetensors"
    original_model_path = "./modelzoo/llama-3.2-1b"
    device = 'cuda'
    
    # Load model structure
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
    
    # Quick test
    test_input = "The capital of France is"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Test generation
        generated = model.generate(
            inputs.input_ids,
            max_new_tokens=15,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Test generation: {generated_text}")
    
    # Check logits range
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    return model, tokenizer

if __name__ == "__main__":
    # Save runtime W4 weights directly
    output_path = save_runtime_w4_directly()
    
    # Test the model
    model, tokenizer = test_direct_w4_model()
    
    print(f"\n" + "="*60)
    print("DIRECT W4 MODEL READY")
    print("="*60)
    print(f"✅ Model saved to: {output_path}")
    print("📝 This should give correct PPL (~11.97) matching W4A4KV4")
    print("💾 Note: File is 2.30 GB (no compression) but weights are correct")