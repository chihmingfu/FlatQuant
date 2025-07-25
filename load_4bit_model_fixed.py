#!/usr/bin/env python3

import torch
import numpy as np
from safetensors import safe_open
import sys
import os
import time
from transformers import AutoTokenizer, LlamaForCausalLM

# Add FlatQuant to path
sys.path.append('/workspace/FlatQuant')
sys.path.append('/workspace/FlatQuant/flatquant')

from save_4bit_weights_packed import RTNQuantizer

class FourBitModelLoader:
    """Loader for 4-bit quantized models saved with bit packing"""
    
    def __init__(self):
        self.quantizer = RTNQuantizer(n_bits=4)
    
    def load_4bit_model(self, model_path_4bit, original_model_path, device='cuda'):
        """
        Load a 4-bit quantized model without runtime quantization
        
        Args:
            model_path_4bit: Path to saved 4-bit model (.safetensors)
            original_model_path: Path to original model (for tokenizer and config)
            device: Device to load model on
        
        Returns:
            model: Loaded model with 4-bit weights reconstructed
            tokenizer: Original tokenizer
        """
        print(f"🔄 Loading 4-bit model from: {model_path_4bit}")
        print(f"📁 Original model path: {original_model_path}")
        print(f"🎯 Target device: {device}")
        
        start_time = time.time()
        
        # Step 1: Load tokenizer and model structure from original
        print("\n📝 Loading tokenizer and model structure...")
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load original model structure (will replace weights)
        model = LlamaForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None  # We'll handle device placement manually
        )
        
        # Step 2: Load and reconstruct 4-bit weights
        print("🔄 Loading and reconstructing 4-bit weights...")
        reconstructed_weights = self._load_and_reconstruct_weights(model_path_4bit)
        
        # Step 3: Replace model weights with reconstructed 4-bit weights
        print("🔧 Replacing model weights...")
        self._replace_model_weights(model, reconstructed_weights)
        
        # Step 4: Move to target device
        print(f"📱 Moving model to {device}...")
        model = model.to(device)
        model.eval()
        
        load_time = time.time() - start_time
        
        # Step 5: Verify model is ready
        print("✅ Model loading completed!")
        print(f"⏱️  Loading time: {load_time:.2f} seconds")
        
        # Quick sanity check
        print("🧪 Running sanity check...")
        self._sanity_check(model, tokenizer, device)
        
        return model, tokenizer
    
    def _load_and_reconstruct_weights(self, model_path_4bit):
        """Load packed 4-bit weights and reconstruct them to BF16"""
        reconstructed_weights = {}
        
        with safe_open(model_path_4bit, framework="pt") as f:
            keys = f.keys()
            
            # Separate regular weights from metadata
            weight_keys = [k for k in keys if not k.endswith("_scale") and not k.endswith("_shape")]
            
            print(f"   Found {len(weight_keys)} weight tensors")
            
            for i, key in enumerate(weight_keys):
                if (i + 1) % 20 == 0:  # Progress indicator
                    print(f"   Processed {i + 1}/{len(weight_keys)} weights...")
                
                scale_key = key + "_scale"
                shape_key = key + "_shape"
                
                if scale_key in keys and shape_key in keys:
                    # This weight was quantized - reconstruct it
                    packed_weight = f.get_tensor(key)
                    scale = f.get_tensor(scale_key).item()
                    original_shape = tuple(f.get_tensor(shape_key).tolist())
                    
                    # Reconstruct the quantized weight
                    reconstructed = self.quantizer.dequantize_tensor(packed_weight, original_shape, scale)
                    reconstructed_weights[key] = reconstructed
                    
                else:
                    # This weight was kept in BF16
                    reconstructed_weights[key] = f.get_tensor(key)
        
        print(f"   ✅ Reconstructed {len(reconstructed_weights)} weights")
        return reconstructed_weights
    
    def _replace_model_weights(self, model, reconstructed_weights):
        """Replace model weights with reconstructed 4-bit weights"""
        replaced_count = 0
        
        # Get all model parameters as a dictionary
        model_state_dict = dict(model.named_parameters())
        
        print(f"   Model has {len(model_state_dict)} parameters")
        print(f"   Reconstructed weights: {len(reconstructed_weights)}")
        
        # Replace parameters that match
        for param_name, param in model_state_dict.items():
            if param_name in reconstructed_weights:
                # Found matching weight
                new_weight = reconstructed_weights[param_name]
                
                # Verify shapes match
                if param.shape != new_weight.shape:
                    raise ValueError(f"Shape mismatch for {param_name}: {param.shape} vs {new_weight.shape}")
                
                # Replace parameter data
                param.data.copy_(new_weight)
                replaced_count += 1
                
        print(f"   ✅ Replaced {replaced_count} model parameters")
        
        # Check for missing weights
        missing_weights = []
        for param_name in model_state_dict.keys():
            if param_name not in reconstructed_weights:
                missing_weights.append(param_name)
        
        if missing_weights:
            print(f"   ⚠️  Missing weights in 4-bit file:")
            for weight in missing_weights[:5]:  # Show first 5
                print(f"      - {weight}")
            if len(missing_weights) > 5:
                print(f"      ... and {len(missing_weights) - 5} more")
        
        # Check for extra weights
        extra_weights = []
        for weight_name in reconstructed_weights.keys():
            if weight_name not in model_state_dict:
                extra_weights.append(weight_name)
        
        if extra_weights:
            print(f"   ⚠️  Extra weights in 4-bit file:")
            for weight in extra_weights[:5]:  # Show first 5
                print(f"      - {weight}")
            if len(extra_weights) > 5:
                print(f"      ... and {len(extra_weights) - 5} more")
        
        return replaced_count
    
    def _sanity_check(self, model, tokenizer, device):
        """Quick sanity check to verify model works"""
        try:
            # Simple forward pass test
            test_input = "Hello, how are you?"
            inputs = tokenizer(test_input, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Check output shape and values
            expected_vocab_size = model.config.vocab_size
            if logits.shape[-1] == expected_vocab_size:
                print("   ✅ Forward pass successful")
                print(f"   📊 Output shape: {logits.shape}")
                print(f"   📈 Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            else:
                print(f"   ❌ Unexpected output shape: {logits.shape}")
                
        except Exception as e:
            print(f"   ❌ Sanity check failed: {str(e)}")
    
    def get_model_info(self, model):
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_usage_mb': memory_mb,
            'memory_usage_gb': memory_mb / 1024,
            'device': next(model.parameters()).device,
            'dtype': next(model.parameters()).dtype
        }
        
        return info

def test_4bit_model_loading():
    """Test function to load and verify 4-bit model"""
    print("🚀 TESTING 4-BIT MODEL LOADING")
    print("="*60)
    
    # Paths
    model_4bit_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b-4bit-packed/model_4bit_packed.safetensors"
    original_model_path = "/workspace/FlatQuant/modelzoo/llama-3.2-1b"
    
    # Create loader
    loader = FourBitModelLoader()
    
    # Load 4-bit model
    model_4bit, tokenizer = loader.load_4bit_model(
        model_4bit_path, 
        original_model_path, 
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Get model info
    info = loader.get_model_info(model_4bit)
    
    print("\n📊 MODEL INFORMATION:")
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Memory usage: {info['memory_usage_gb']:.2f} GB")
    print(f"Device: {info['device']}")
    print(f"Dtype: {info['dtype']}")
    
    # Test generation
    print("\n🎯 TESTING TEXT GENERATION:")
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "Once upon a time"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model_4bit.device)
            
            with torch.no_grad():
                outputs = model_4bit.generate(
                    inputs.input_ids,
                    max_new_tokens=15,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            
        except Exception as e:
            print(f"Generation failed: {str(e)}")
    
    print("\n✅ 4-bit model loading test completed!")
    return model_4bit, tokenizer

if __name__ == "__main__":
    # Test the 4-bit model loading
    model, tokenizer = test_4bit_model_loading()
    
    print("\n🎯 READY FOR PPL VALIDATION")
    print("Model loaded and ready for perplexity testing")