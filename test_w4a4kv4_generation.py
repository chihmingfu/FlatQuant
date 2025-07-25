#!/usr/bin/env python3

import subprocess
import sys
import os

def run_w4a4kv4_generation_test():
    print("=" * 60)
    print("Testing LLaMA-3.2-1B W4A4KV4 Text Generation Quality")
    print("=" * 60)
    
    # Read original main.py
    with open('main.py', 'r') as f:
        main_content = f.read()
    
    # Find insertion point (right after model evaluation before lm_eval)
    insertion_point = main_content.find("# Evaluating PPL")
    
    if insertion_point == -1:
        print("ERROR: Could not find insertion point in main.py")
        return
    
    # Create generation test code that will run in main.py context
    generation_test = '''
    # === GENERATION TEST START ===
    import torch
    print("\\n" + "=" * 60)
    print("Testing W4A4KV4 Text Generation Quality")
    print("=" * 60)
    
    # Test prompts for generation quality
    test_prompts = [
        "The capital of France is",
        "Machine learning is",
        "Once upon a time",
        "Hello, my name is",
        "In 2025, artificial intelligence",
    ]
    
    model.eval()  # Ensure model is in eval mode
    
    for i, prompt in enumerate(test_prompts):
        print(f"\\nTest {i+1}: '{prompt}'")
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(utils.DEV)
            
            # Generate text
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else 2,
                    repetition_penalty=1.1
                )
                
                # Decode generated text
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Generated: {generated_text}")
                
                # Quality check
                input_length = input_ids.shape[1]
                new_tokens = outputs[0][input_length:]
                unique_new_tokens = len(set(new_tokens.tolist()))
                
                if unique_new_tokens < 3:
                    print("⚠️  Warning: Low token diversity - possible repetition issue")
                elif len(generated_text.strip()) > len(prompt.strip()) + 5:
                    print("✅ Generation appears coherent")
                else:
                    print("⚠️  Warning: Very short generation")
                    
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("-" * 40)
    
    print("\\n" + "=" * 60)
    print("W4A4KV4 Generation test completed!")
    print("=" * 60)
    
    # Exit after generation test to avoid running evaluations
    import sys
    sys.exit(0)
    # === GENERATION TEST END ===
    
    '''
    
    # Insert the generation test code
    modified_main = main_content[:insertion_point] + generation_test + "\n    " + main_content[insertion_point:]
    
    # Write the modified main.py
    with open('main_with_w4a4kv4_gen_test.py', 'w') as f:
        f.write(modified_main)
    
    print("Running W4A4KV4 quantized model with generation test...")
    
    # Run the modified main with our quantization settings
    cmd = [
        'python', 'main_with_w4a4kv4_gen_test.py',
        '--model', './modelzoo/llama-3.2-1b',
        '--w_bits', '4', '--a_bits', '4',
        '--k_bits', '4', '--k_asym', '--k_groupsize', '128',
        '--v_bits', '4', '--v_asym', '--v_groupsize', '128',
        '--cali_bsz', '4', '--epoch', '15', '--flat_lr', '5e-3',
        '--lwc', '--lac', '--cali_trans', '--add_diag',
        '--output_dir', './outputs',
        '--reload_matrix', '--matrix_path', './outputs/llama-3.2-1b/w4a4/exp'
    ]
    
    try:
        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("=" * 60)
        print("GENERATION TEST OUTPUT:")
        print("=" * 60)
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nTest completed with return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("ERROR: Test timed out after 5 minutes")
    except Exception as e:
        print(f"ERROR: Test failed with exception: {str(e)}")
    finally:
        # Clean up
        if os.path.exists('main_with_w4a4kv4_gen_test.py'):
            os.remove('main_with_w4a4kv4_gen_test.py')

if __name__ == "__main__":
    run_w4a4kv4_generation_test()