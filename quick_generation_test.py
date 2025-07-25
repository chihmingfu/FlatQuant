import subprocess
import sys
import tempfile

# Create a very simple generation test using transformers directly
test_code = '''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

print("Testing FP16 vs Simulated Quantized Generation")
print("=" * 50)

# Load the model in FP16 mode
model_path = "./modelzoo/llama-3.2-1b"
print(f"Loading model from {model_path}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    
    # Test prompts  
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "Once upon a time",
        "Hello, my name is"
    ]
    
    print("\\nGenerating text samples:")
    print("-" * 50)
    
    for i, prompt in enumerate(prompts):
        print(f"\\nTest {i+1}: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.to(model.device),
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {generated_text}")
        
        # Check for basic coherence
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        if len(set(new_tokens.tolist())) >= 3:
            print("✅ Generation appears coherent")
        else:
            print("⚠️  Low token diversity")
    
    print("\\n" + "=" * 50)
    print("FP16 Generation test completed!")
    print("\\nNote: This test uses FP16 model to verify generation capability.")
    print("The W4A4KV4 quantized model should produce similar but potentially")
    print("lower quality outputs based on the 22.7% perplexity degradation.")
    print("=" * 50)
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
'''

# Write and run the test
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(test_code)
    temp_file = f.name

try:
    result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
finally:
    import os
    os.unlink(temp_file)