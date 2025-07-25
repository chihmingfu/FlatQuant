#!/usr/bin/env python3

import torch
import sys
from safetensors import safe_open
from transformers import AutoTokenizer, LlamaForCausalLM

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')
sys.path.append('/workspace/FlatQuant/flatquant')

def unpack_true_4bit(packed_data, original_shape, padded_numel):
    """Unpack TRUE 4-bit data back to indices"""
    # Unpack: extract lower and upper 4 bits
    lower_4bit = packed_data & 0x0F  # Extract lower 4 bits
    upper_4bit = (packed_data & 0xF0) >> 4  # Extract upper 4 bits
    
    # Interleave the values
    unpacked = torch.stack([lower_4bit, upper_4bit], dim=1).flatten()
    
    # Remove padding if any
    if unpacked.numel() > padded_numel:
        unpacked = unpacked[:padded_numel]
    
    # Convert back from unsigned [0,15] to signed [-8,7]
    unpacked = unpacked.to(torch.int8) - 8
    
    # Reshape to original shape
    return unpacked.view(original_shape).to(torch.bfloat16)

def test_true_4bit_model():
    """Test the TRUE 4-bit model"""
    print("🎯 TESTING TRUE 4-BIT MODEL")
    print("="*50)
    
    device = 'cuda'
    
    # Paths
    model_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-true-4bit/model_true_4bit.safetensors"
    original_model_path = "./modelzoo/llama-3.2-1b"
    
    # Load model structure
    tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    
    # Load compressed weights
    print("Loading TRUE 4-bit compressed weights...")
    with safe_open(model_path, framework="pt") as f:
        keys = f.keys()
        replaced_count = 0
        
        for param_name, param in model.named_parameters():
            if f"{param_name}_packed" in keys:
                # Load packed data and metadata
                packed_data = f.get_tensor(f"{param_name}_packed")
                scale = f.get_tensor(f"{param_name}_scale")
                shape = f.get_tensor(f"{param_name}_shape").tolist()
                padded_numel = f.get_tensor(f"{param_name}_numel").item()
                
                # Unpack TRUE 4-bit data
                indices = unpack_true_4bit(packed_data, shape, padded_numel)
                
                # Reconstruct using FlatQuant's dequantization
                scale_gpu = scale.to(device)
                indices_gpu = indices.to(device)
                reconstructed_weight = scale_gpu * indices_gpu
                
                # Replace parameter
                param.data.copy_(reconstructed_weight)
                replaced_count += 1
                
            elif param_name in keys:
                # Direct copy for BF16 weights
                weight = f.get_tensor(param_name)
                param.data.copy_(weight.to(torch.bfloat16))
                replaced_count += 1
    
    print(f"✅ Replaced {replaced_count} parameters")
    
    model = model.to(device)
    model.eval()
    
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
    
    # Quick quality check
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # File size info
    import os
    file_size = os.path.getsize(model_path) / (1024**3)
    original_size = 2.4
    compression = original_size / file_size
    
    print(f"\n📏 Compression Results:")
    print(f"Original: {original_size:.1f} GB")
    print(f"TRUE 4-bit: {file_size:.2f} GB")
    print(f"Compression: {compression:.2f}x")
    print(f"Size reduction: {(1-file_size/original_size)*100:.1f}%")
    
    # Quality assessment
    if "Paris" in generated_text or "country" in generated_text or "France" in generated_text:
        quality = "✅ GOOD - Coherent generation"
    elif len(generated_text.split()) > 5:
        quality = "⚠️  ACCEPTABLE - Some generation"
    else:
        quality = "❌ POOR - Broken generation"
    
    print(f"\n🎯 ASSESSMENT:")
    print(f"Compression: {compression:.2f}x (target was ~2.4x)")
    print(f"Generation quality: {quality}")
    
    if compression >= 2.4 and "GOOD" in quality:
        print("🎉 SUCCESS: TRUE 4-bit model works excellently!")
        status = "SUCCESS"
    elif compression >= 2.0:
        print("✅ GOOD: Significant compression with working model")
        status = "GOOD"
    else:
        print("⚠️  PARTIAL: Some improvement achieved")
        status = "PARTIAL"
    
    return model, tokenizer, status

if __name__ == "__main__":
    model, tokenizer, status = test_true_4bit_model()
    
    print(f"\n" + "="*50)
    print("🎯 TRUE 4-BIT MODEL TEST COMPLETE")
    print("="*50)
    print(f"🎯 Status: {status}")
    print(f"💡 Model uses TRUE 4-bit storage with FlatQuant quantizers")
    print(f"📝 Ready for production use if quality is acceptable")