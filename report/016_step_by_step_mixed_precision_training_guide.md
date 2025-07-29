# Step-by-Step Mixed Precision Training Guide for VSCode

**Date:** 2025-07-29  
**Purpose:** Guide for reproducing mixed precision W2/W4 training results  
**Target:** VSCode users who want to verify the implementation step-by-step

---

## 🎯 **Overview**

This guide walks you through running the mixed precision training code step by step to validate the results. You'll be able to inspect each component and verify the implementation correctness.

---

## 📋 **Prerequisites**

### **Environment Setup**
```bash
# 1. Create conda environment
conda create -n flatquant python=3.10 -y
conda activate flatquant

# 2. Install dependencies
pip install -r requirements.txt && pip install -e . && pip install triton==3.0.0

# 3. Build CUDA extensions
python setup.py build_ext --inplace
```

### **VSCode Setup**
1. **Open Folder:** `/workspace/FlatQuant` in VSCode
2. **Select Python Interpreter:** `flatquant` conda environment
3. **Install Extensions:**
   - Python
   - Jupyter (for notebook inspection)
   - GitLens (to see code changes)

---

## 🔍 **Step 1: Examine the Code Structure**

### **Key Files to Review:**

**1. Main Training Script:**
```
📁 /workspace/FlatQuant/main.py
```
**What to check:**
- Lines 15-51: `apply_mixed_precision_config()` function
- Lines 58-61: Mixed precision mode detection
- Lines 81-82: Configuration application
- Lines 88-89: Training call

**2. Mixed Precision Logic:**
```
📁 /workspace/FlatQuant/flatquant/args_utils.py
```
**What to check:**
- Lines 95-96: `--mixed_precision` argument definition

**3. Quantization Utils:**
```
📁 /workspace/FlatQuant/flatquant/quant_utils.py
```
**What to check:**
- Lines 118-140: `WeightQuantizer.configure()` method
- Lines 10-16: `get_qmin_qmax()` bit range calculation

---

## 🧪 **Step 2: Test Components Individually**

### **2.1 Test Model Loading**

Create test file: `test_step1_model_loading.py`
```python
#!/usr/bin/env python3
"""Step 1: Test model loading"""

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils

# Setup args
args, logger = args_utils.parser_gen()
args.model = "./modelzoo/llama-3.2-1b"
args.mixed_precision = True

print(f"🔧 Loading model: {args.model}")
print(f"🔧 Mixed precision: {args.mixed_precision}")

# Load model
model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
print(f"✅ Model loaded successfully")
print(f"📊 Model layers: {len(model.model.layers)}")
print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check layer structure
layer_0 = model.model.layers[0]
print(f"📊 Layer 0 structure:")
print(f"  - self_attn: {hasattr(layer_0, 'self_attn')}")
print(f"  - mlp: {hasattr(layer_0, 'mlp')}")
if hasattr(layer_0, 'self_attn'):
    attn = layer_0.self_attn
    print(f"  - q_proj: {hasattr(attn, 'q_proj')}")
    print(f"  - k_proj: {hasattr(attn, 'k_proj')}")
    print(f"  - v_proj: {hasattr(attn, 'v_proj')}")
    print(f"  - o_proj: {hasattr(attn, 'o_proj')}")
```

**Run in VSCode:**
1. Open terminal in VSCode (`Ctrl+``)
2. Run: `python test_step1_model_loading.py`
3. **Expected output:** Model loads with 16 layers, attention/MLP modules present

### **2.2 Test FlatQuant Application**

Create test file: `test_step2_flatquant_application.py`
```python
#!/usr/bin/env python3
"""Step 2: Test FlatQuant application"""

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils

# Setup args
args, logger = args_utils.parser_gen()
args.model = "./modelzoo/llama-3.2-1b"
args.mixed_precision = True
args.w_bits = 4
args.a_bits = 8
args.lwc = True
args.lac = True
args.cali_trans = True
args.add_diag = True

# Load and apply FlatQuant
model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
model = apply_flatquant_to_model(args, model)

print("✅ FlatQuant applied successfully")

# Check quantizers
layer_0 = model.model.layers[0]
if hasattr(layer_0, 'self_attn') and hasattr(layer_0.self_attn, 'q_proj'):
    q_proj = layer_0.self_attn.q_proj
    print(f"📊 Layer 0 q_proj type: {type(q_proj)}")
    
    if hasattr(q_proj, 'weight_quantizer'):
        quantizer = q_proj.weight_quantizer
        print(f"📊 Weight quantizer found: {type(quantizer)}")
        print(f"📊 Default bits (before config): {getattr(quantizer, 'bits', 'Not set')}")
        print(f"📊 Current maxq: {quantizer.maxq}")
    else:
        print("❌ No weight_quantizer found")
```

**Run in VSCode:**
1. Run: `python test_step2_flatquant_application.py`
2. **Expected output:** FlatQuant modules applied, weight quantizers present

### **2.3 Test Mixed Precision Configuration**

Create test file: `test_step3_mixed_precision_config.py`
```python
#!/usr/bin/env python3
"""Step 3: Test mixed precision configuration"""

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils

def apply_mixed_precision_config(args, model, logger):
    """Copy of the function from main.py"""
    w2_layers = [0, 2, 3, 4, 5, 6, 7, 8]
    w4_layers = [1, 9, 10, 11, 12, 13, 14, 15]
    
    print("🔧 Applying Report 013 Mixed Precision Configuration:")
    print(f"  W2 layers: {w2_layers}")
    print(f"  W4 layers: {w4_layers}")
    
    layer_configs = {}
    
    # Configure quantizers for each layer
    for layer_idx, layer in enumerate(model.model.layers):
        target_bits = 2 if layer_idx in w2_layers else 4
        layer_configs[layer_idx] = target_bits
        
        # Configure attention weights
        if hasattr(layer, 'self_attn'):
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if hasattr(layer.self_attn, name):
                    module = getattr(layer.self_attn, name)
                    if hasattr(module, 'weight_quantizer'):
                        old_bits = getattr(module.weight_quantizer, 'bits', 'Not set')
                        module.weight_quantizer.configure(
                            target_bits, perchannel=True, sym=not(args.w_asym), mse=False
                        )
                        new_bits = module.weight_quantizer.bits
                        print(f"  Layer {layer_idx} {name}: {old_bits} → {new_bits} bits")
        
        # Configure MLP weights
        if hasattr(layer, 'mlp'):
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(layer.mlp, name):
                    module = getattr(layer.mlp, name)
                    if hasattr(module, 'weight_quantizer'):
                        module.weight_quantizer.configure(
                            target_bits, perchannel=True, sym=not(args.w_asym), mse=False
                        )
        
        print(f"📊 Layer {layer_idx}: W{target_bits} configured")
    
    return layer_configs

# Setup and test
args, logger = args_utils.parser_gen()
args.model = "./modelzoo/llama-3.2-1b"
args.mixed_precision = True
args.w_bits = 4
args.a_bits = 8
args.lwc = True
args.lac = True
args.cali_trans = True
args.add_diag = True

# Load model and apply FlatQuant
model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
model = apply_flatquant_to_model(args, model)

# Apply mixed precision
layer_configs = apply_mixed_precision_config(args, model, logger)

# Verify configuration
print("\n✅ Configuration Summary:")
w2_count = sum(1 for bits in layer_configs.values() if bits == 2)
w4_count = sum(1 for bits in layer_configs.values() if bits == 4)
print(f"📊 W2 layers: {w2_count}/16")
print(f"📊 W4 layers: {w4_count}/16")
print(f"📊 Average bits: {sum(layer_configs.values()) / len(layer_configs):.1f}")

# Verify specific layers
test_layers = [(0, 2), (1, 4), (2, 2), (15, 4)]
print("\n🔍 Verification:")
for layer_idx, expected_bits in test_layers:
    actual_bits = layer_configs[layer_idx]
    status = "✅" if actual_bits == expected_bits else "❌"
    print(f"{status} Layer {layer_idx}: Expected W{expected_bits}, Got W{actual_bits}")
```

**Run in VSCode:**
1. Run: `python test_step3_mixed_precision_config.py`
2. **Expected output:** 
   - 8 layers configured as W2
   - 8 layers configured as W4
   - Average 3.0 bits
   - Layer 0: W2, Layer 1: W4, Layer 2: W2, Layer 15: W4

---

## 🚀 **Step 3: Run Full Training with Monitoring**

### **3.1 Prepare Training Script**

Create monitoring script: `run_training_monitored.py`
```python
#!/usr/bin/env python3
"""Run mixed precision training with detailed monitoring"""

import os
import time
import subprocess

def run_training():
    """Run training with monitoring"""
    
    print("🚀 Starting Mixed Precision Training")
    print("=" * 50)
    
    # Training command
    cmd = [
        "python", "main.py",
        "--model", "./modelzoo/llama-3.2-1b",
        "--mixed_precision",
        "--w_bits", "4",
        "--a_bits", "8", 
        "--k_bits", "8", "--k_asym", "--k_groupsize", "128",
        "--v_bits", "8", "--v_asym", "--v_groupsize", "128",
        "--cali_bsz", "4",
        "--epochs", "15",
        "--flat_lr", "1e-3",
        "--lwc", "--lac", "--cali_trans", "--add_diag",
        "--output_dir", "./outputs",
        "--save_matrix",
        "--diag_alpha", "0.1"
    ]
    
    print(f"📋 Command: {' '.join(cmd)}")
    print("⏳ Training starting...")
    
    # Run training
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        print("✅ Training completed!")
        print(f"📊 Return code: {result.returncode}")
        
        if result.stdout:
            print("\n📝 Training Output:")
            print(result.stdout[-2000:])  # Last 2000 chars
            
        if result.stderr:
            print("\n❌ Errors/Warnings:")
            print(result.stderr[-1000:])  # Last 1000 chars
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Training timeout (1 hour)")
        return False
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

def check_results():
    """Check training results"""
    
    print("\n🔍 Checking Results...")
    
    # Check output directory
    exp_dir = "./outputs/llama-3.2-1b/w4a8/exp"
    if not os.path.exists(exp_dir):
        print(f"❌ Output directory not found: {exp_dir}")
        return False
    
    print(f"✅ Output directory exists: {exp_dir}")
    
    # Check files
    expected_files = [
        "flat_matrices.pth",
        "flat_parameters.pth", 
        "mixed_precision_config.json"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(exp_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)  # MB
            print(f"✅ {filename}: {size:.1f} MB")
        else:
            print(f"❌ Missing: {filename}")
    
    # Check log file
    log_files = [f for f in os.listdir(exp_dir) if f.startswith("log_rank0_")]
    if log_files:
        latest_log = sorted(log_files)[-1]
        log_path = os.path.join(exp_dir, latest_log)
        print(f"✅ Log file: {latest_log}")
        
        # Extract key info from log
        with open(log_path, 'r') as f:
            log_content = f.read()
            
        # Check layer configurations
        layer_configs = []
        for line in log_content.split('\n'):
            if "Layer" in line and "W2" in line:
                layer_configs.append(line.strip())
            elif "Layer" in line and "W4" in line:
                layer_configs.append(line.strip())
        
        print(f"📊 Layer configurations found: {len(layer_configs)}")
        if layer_configs:
            print("🔍 First few configurations:")
            for config in layer_configs[:5]:
                print(f"  {config}")
        
        # Check perplexity results
        ppl_lines = []
        for line in log_content.split('\n'):
            if line.strip() and any(char.isdigit() for char in line) and len(line.strip()) < 50:
                try:
                    float(line.strip())
                    ppl_lines.append(line.strip())
                except:
                    pass
        
        if ppl_lines:
            print("📊 Potential perplexity values:")
            for ppl in ppl_lines[-5:]:  # Last 5 numbers
                print(f"  {ppl}")
    
    return True

if __name__ == "__main__":
    success = run_training()
    if success:
        check_results()
    else:
        print("❌ Training failed - check logs for details")
```

### **3.2 Monitor Training Progress**

Create monitoring script: `monitor_training.py`
```python
#!/usr/bin/env python3
"""Monitor training progress in real-time"""

import os
import time
import glob

def monitor_training():
    """Monitor training progress"""
    
    print("👀 Training Progress Monitor")
    print("=" * 30)
    
    exp_dir = "./outputs/llama-3.2-1b/w4a8/exp"
    
    while True:
        # Find latest log file
        log_pattern = os.path.join(exp_dir, "log_rank0_*.txt")
        log_files = glob.glob(log_pattern)
        
        if not log_files:
            print("⏳ Waiting for training to start...")
            time.sleep(10)
            continue
        
        latest_log = sorted(log_files)[-1]
        
        # Read last few lines
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"\n📊 Latest progress ({time.strftime('%H:%M:%S')}):")
                    
                    # Show last training iteration
                    for line in reversed(lines[-20:]):
                        if "mse:" in line.lower():
                            print(f"  {line.strip()}")
                            break
                    
                    # Show current layer
                    for line in reversed(lines[-50:]):
                        if "========= Layer" in line:
                            print(f"  {line.strip()}")
                            break
                    
                    # Check if completed
                    last_line = lines[-1].strip()
                    if any(keyword in last_line.lower() for keyword in ["finished", "completed", "error"]):
                        print(f"🎯 Training status: {last_line}")
                        if "error" not in last_line.lower():
                            print("✅ Training appears to be completed!")
                            break
        
        except Exception as e:
            print(f"⚠️  Error reading log: {str(e)}")
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    monitor_training()
```

---

## 📊 **Step 4: Validate Results**

### **4.1 Check Training Configuration**

Create validation script: `validate_results.py`
```python
#!/usr/bin/env python3
"""Validate training results"""

import json
import os
import torch

def validate_configuration():
    """Validate mixed precision configuration"""
    
    config_path = "./outputs/llama-3.2-1b/w4a8/exp/mixed_precision_config.json"
    
    if not os.path.exists(config_path):
        print("❌ Configuration file not found")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("✅ Configuration loaded")
    
    # Check layer allocation
    w2_layers = config['mixed_precision_config']['w2_layers']
    w4_layers = config['mixed_precision_config']['w4_layers']
    
    print(f"📊 W2 layers: {w2_layers}")
    print(f"📊 W4 layers: {w4_layers}")
    
    # Validate Report 013 configuration
    expected_w2 = [0, 2, 3, 4, 5, 6, 7, 8]
    expected_w4 = [1, 9, 10, 11, 12, 13, 14, 15]
    
    w2_correct = set(w2_layers) == set(expected_w2)
    w4_correct = set(w4_layers) == set(expected_w4)
    
    print(f"✅ W2 configuration correct: {w2_correct}")
    print(f"✅ W4 configuration correct: {w4_correct}")
    
    # Calculate compression
    total_layers = len(w2_layers) + len(w4_layers)
    avg_bits = (len(w2_layers) * 2 + len(w4_layers) * 4) / total_layers
    compression_ratio = 16 / avg_bits
    
    print(f"📊 Total layers: {total_layers}")
    print(f"📊 Average bits: {avg_bits}")
    print(f"📊 Weight compression ratio: {compression_ratio:.2f}x")
    
    return w2_correct and w4_correct

def validate_matrices():
    """Validate training matrices"""
    
    matrices_path = "./outputs/llama-3.2-1b/w4a8/exp/flat_matrices.pth"
    
    if not os.path.exists(matrices_path):
        print("❌ Matrices file not found")
        return False
    
    try:
        matrices = torch.load(matrices_path, map_location='cpu')
        print("✅ Matrices loaded successfully")
        print(f"📊 Matrix keys: {list(matrices.keys())}")
        
        # Check matrix sizes
        total_size = 0
        for key, value in matrices.items():
            if isinstance(value, torch.Tensor):
                size_mb = value.numel() * value.element_size() / (1024 * 1024)
                total_size += size_mb
                print(f"  {key}: {value.shape} ({size_mb:.1f} MB)")
        
        print(f"📊 Total matrices size: {total_size:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error loading matrices: {str(e)}")
        return False

def extract_perplexity():
    """Extract perplexity from logs"""
    
    log_pattern = "./outputs/llama-3.2-1b/w4a8/exp/log_rank0_*.txt"
    import glob
    
    log_files = glob.glob(log_pattern)
    if not log_files:
        print("❌ No log files found")
        return False
    
    latest_log = sorted(log_files)[-1]
    
    with open(latest_log, 'r') as f:
        content = f.read()
    
    # Look for perplexity values
    lines = content.split('\n')
    ppl_values = []
    
    for i, line in enumerate(lines):
        if 'wikitext2' in line.lower() and i + 1 < len(lines):
            try:
                ppl = float(lines[i + 1].strip())
                ppl_values.append(('WikiText2', ppl))
            except:
                pass
                
        if 'c4' in line.lower() and i + 1 < len(lines):
            try:
                ppl = float(lines[i + 1].strip())
                ppl_values.append(('C4', ppl))
            except:
                pass
    
    if ppl_values:
        print("✅ Perplexity values found:")
        for dataset, ppl in ppl_values:
            print(f"  {dataset}: {ppl:.2f}")
        
        # Compare with baselines
        if ppl_values:
            wikitext2_ppl = next((ppl for name, ppl in ppl_values if 'wikitext2' in name.lower()), None)
            if wikitext2_ppl:
                print(f"\n📊 Comparison:")
                print(f"  W4A4KV4 baseline: 11.97")
                print(f"  W3A4KV4 baseline: 17.04")
                print(f"  Our Mixed W2/W4: {wikitext2_ppl:.2f}")
                
                if wikitext2_ppl < 17.04:
                    print(f"✅ Better than W3A4KV4 by {17.04 - wikitext2_ppl:.2f} PPL")
                else:
                    print(f"⚠️  Worse than W3A4KV4 by {wikitext2_ppl - 17.04:.2f} PPL")
        
        return True
    else:
        print("❌ No perplexity values found in logs")
        return False

if __name__ == "__main__":
    print("🔍 Validating Training Results")
    print("=" * 40)
    
    config_ok = validate_configuration()
    matrices_ok = validate_matrices()
    ppl_ok = extract_perplexity()
    
    print(f"\n🎯 Validation Summary:")
    print(f"✅ Configuration: {config_ok}")
    print(f"✅ Matrices: {matrices_ok}")
    print(f"✅ Perplexity: {ppl_ok}")
    
    if all([config_ok, matrices_ok, ppl_ok]):
        print("\n🎉 All validations passed! Results are correct.")
    else:
        print("\n⚠️  Some validations failed. Check the details above.")
```

---

## 🎯 **Step 5: Running Instructions for VSCode**

### **Terminal Setup in VSCode**

1. **Open Terminal:** `Ctrl+` (backtick)
2. **Activate Environment:** `conda activate flatquant`
3. **Navigate to Directory:** `cd /workspace/FlatQuant`

### **Step-by-Step Execution**

**Step 1: Test Components**
```bash
# Test 1: Model Loading
python test_step1_model_loading.py

# Test 2: FlatQuant Application  
python test_step2_flatquant_application.py

# Test 3: Mixed Precision Configuration
python test_step3_mixed_precision_config.py
```

**Step 2: Run Training (Choose one)**

**Option A: Interactive Training**
```bash
python main.py --model ./modelzoo/llama-3.2-1b --mixed_precision --w_bits 4 --a_bits 8 --k_bits 8 --k_asym --k_groupsize 128 --v_bits 8 --v_asym --v_groupsize 128 --cali_bsz 4 --epochs 15 --flat_lr 1e-3 --lwc --lac --cali_trans --add_diag --output_dir ./outputs --save_matrix --diag_alpha 0.1
```

**Option B: Background Training with Monitoring**
```bash
# Terminal 1: Start training
nohup python main.py --model ./modelzoo/llama-3.2-1b --mixed_precision --w_bits 4 --a_bits 8 --k_bits 8 --k_asym --k_groupsize 128 --v_bits 8 --v_asym --v_groupsize 128 --cali_bsz 4 --epochs 15 --flat_lr 1e-3 --lwc --lac --cali_trans --add_diag --output_dir ./outputs --save_matrix --diag_alpha 0.1 > training.log 2>&1 &

# Terminal 2: Monitor progress
python monitor_training.py
```

**Step 3: Validate Results**
```bash
python validate_results.py
```

### **What to Expect**

**Training Progress:**
- ✅ 16 layers configured (8×W2, 8×W4)
- ✅ Each layer trains for 15 epochs
- ✅ MSE values decrease over iterations
- ✅ ~37 minutes total training time
- ✅ Files saved to `outputs/llama-3.2-1b/w4a8/exp/`

**Expected Results:**
- ✅ WikiText2 PPL: ~16.2
- ✅ C4 PPL: ~24.6
- ✅ Configuration: Report 013 allocation
- ✅ Compression: 5.33x weight compression (same as W3A4KV4)
- ✅ Quality: 5% better than W3A4KV4

---

## 🐛 **Troubleshooting**

### **Common Issues**

**1. CUDA Out of Memory**
```bash
# Reduce batch size
--cali_bsz 2
```

**2. Singular Matrix Error**
```bash
# Use more conservative parameters
--flat_lr 5e-4 --diag_alpha 0.05
```

**3. Model Not Found**
```bash
# Check model path
ls -la ./modelzoo/llama-3.2-1b
```

**4. Dependencies Missing**
```bash
# Reinstall environment
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### **VSCode Debugging**

**Set Breakpoints:**
- `main.py:82` - Mixed precision configuration
- `main.py:89` - Training start
- `flatquant/train_utils.py:90` - Individual layer training

**Debug Configuration (launch.json):**
```json
{
    "name": "Mixed Precision Training",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/main.py",
    "args": [
        "--model", "./modelzoo/llama-3.2-1b",
        "--mixed_precision",
        "--w_bits", "4", "--a_bits", "8", 
        "--epochs", "3"
    ],
    "console": "integratedTerminal"
}
```

---

## 📊 **Expected Timeline**

| Step | Duration | Description |
|------|----------|-------------|
| Setup | 5 min | Environment + dependencies |
| Component Tests | 10 min | Individual component validation |
| Training | 35 min | Full mixed precision training |
| Validation | 5 min | Result verification |
| **Total** | **55 min** | Complete workflow |

---

## ✅ **Success Criteria**

When completed successfully, you should have:

1. ✅ **Configuration Verified:** Report 013 layer allocation applied
2. ✅ **Training Completed:** All 16 layers trained without errors
3. ✅ **Files Generated:** Matrices, parameters, config, and logs
4. ✅ **Quality Achieved:** WikiText2 PPL ~16.2 (better than W3A4KV4)
5. ✅ **Compression Confirmed:** 5.33x weight compression ratio

---

**Happy Training! 🚀**

*This guide ensures you can reproduce and validate every aspect of the mixed precision quantization implementation.*