# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation and Setup
```bash
# Create conda environment and install dependencies
conda create -n flatquant python=3.10 -y
conda activate flatquant
pip install -r requirements.txt && pip install -e . && pip install triton==3.0.0

# For LLaMA-3.1 or Qwen-2.5, use transformers==4.45.0 instead
```

### Building CUDA Extensions
```bash
# The setup.py automatically builds CUDA extensions and third-party dependencies
python setup.py build_ext --inplace
```

### Running FlatQuant Quantization
```bash
# W4A4KV4 quantization example (LLaMA-3-8B)
python ./main.py \
    --model ./modelzoo/llama-3/llama-3-8b \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --save_matrix \
    --lm_eval --lm_eval_batch_size 16

# For DeepSeek V3/R1, use main_dpskv3.py instead and add --v3_not_last flag
python ./main_dpskv3.py \
    --model ./modelzoo/deepseek-v3 \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --v3_not_last \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --save_matrix
```

### Using Pre-trained Matrices
```bash
# Load existing quantized model without re-training
python ./main.py \
    --model ./modelzoo/llama-3/llama-3-8b \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --matrix_path ./outputs/llama-3-8b/w4a4/exp/flat_matrices.pth \
    --lm_eval --lm_eval_batch_size 16
```

### Benchmarking
```bash
# Run kernel benchmarks
python ./benchmarks/kernel_benchmark.py
python ./benchmarks/layer_benchmark.py
python ./benchmarks/qlinear_benchmark.py
python ./benchmarks/qattention_benchmark.py
```

### Flatness Visualization
```bash
# Plot weight/activation flatness analysis
python ./plot_flatness.py \
    --model ./modelzoo/llama-3/llama-3-8b \
    --distribute_model --add_diag \
    --matrix_path ./modelzoo/flatquant/llama-3-8b/w4a4
```

## Architecture Overview

FlatQuant is a PyTorch-based quantization framework for Large Language Models that uses Fast and Learnable Affine Transformations to achieve high-accuracy low-bit quantization.

### Core Components

1. **flatquant/** - Main quantization package
   - `flat_linear.py` - Custom linear layers with flat transformations (AddDiag, HadamardTransform)
   - `quant_utils.py` - Quantization primitives (asymmetric/symmetric quantizers, group-wise quantization)
   - `train_utils.py` - Calibration and training utilities for learning transformations
   - `flat_utils.py` - Utility functions for flat transformation matrix operations
   - `args_utils.py` - Command-line argument parsing and configuration management
   - `model_tools/` - Model-specific implementations:
     - `llama_utils.py` - LLaMA 2/3 integration with FlatQuantLlamaMLP and FlatQuantLlamaAttention
     - `llama31_utils.py` - LLaMA 3.1 specific support
     - `qwen_utils.py` - Qwen-2.5 integration
     - `deepseekv3_utils.py` - DeepSeek V3/R1 integration

2. **deploy/** - Optimized deployment code
   - `kernels/` - Custom CUDA kernels for efficient W4A4 operations:
     - `gemm.cu` - Optimized GEMM operations for quantized inference
     - `quant.cu` - Quantization/dequantization kernels
     - `flashinfer.cu` - FlashAttention-style optimized attention kernels
     - `kron_matmul.py` - Kronecker product matrix multiplication
     - `block_matmul.py` - Block-wise matrix operations
   - `nn/` - Deployment-ready neural network modules:
     - `linear.py` - Linear4bit layer for efficient inference
     - `quantization.py` - Production quantization utilities
     - `online_trans.py` - Online transformation operations

3. **vllm_custom/** - vLLM integration for fake quantized inference
   - `registry.py` - Registration system for fake quantized models in vLLM
   - `llama_fake_quantized.py` - LLaMA fake quantization implementation
   - `qwen2_fake_quantized.py` - Qwen2 fake quantization implementation
   - Model-specific FlatQuant implementations for vLLM compatibility

### Key Design Patterns

- **Transformation Types**: Uses AddDiag (learnable diagonal) and Hadamard transformations to flatten weights/activations
- **Quantization Schemes**: Supports both RTN (Round-to-Nearest) and GPTQ weight quantization
- **Group-wise Quantization**: Configurable group sizes for weights, activations, and KV-cache
- **Asymmetric Quantization**: Optional for KV-cache to improve accuracy

### Model Integration Flow

1. Models are wrapped with FlatQuant-specific layers in `model_tools/`
2. Calibration learns optimal transformations using a small dataset (WikiText2, C4, or Pile)
3. Quantized models can be evaluated on perplexity and QA tasks
4. Pre-trained transformation matrices can be saved/loaded for reproducibility

### Entry Points

- **`main.py`** - Primary quantization pipeline for standard models (LLaMA-2/3, Qwen-2.5)
- **`main_dpskv3.py`** - Specialized entry point for DeepSeek V3/R1 models
- **`plot_flatness.py`** - Visualization tool for weight/activation flatness analysis

### Supported Models and Configurations

**Models:**
- LLaMA-2 (7B, 13B, 70B)
- LLaMA-3 (8B, 70B)  
- LLaMA-3.1 (8B, 70B)
- Qwen-2.5-Instruct (7B, 32B)
- DeepSeek V3/R1

**Quantization Modes:**
- W4A4KV4: 4-bit weights, 4-bit activations, 4-bit KV-cache
- W4A16: 4-bit weights only
- Configurable group sizes and asymmetric quantization

### Output Structure

Quantized models are saved to `./outputs/{model_name}/{quantization_config}/exp/`:
- `flat_matrices.pth` - Pre-trained transformation matrices
- `flat_parameters.pth` - Training parameters and configurations
- Log files with perplexity results and training progress

### Third-party Dependencies

- **CUTLASS**: NVIDIA's CUDA template library for optimized GEMM operations
- **fast-hadamard-transform**: Efficient Hadamard transform implementation
- Built via CMake during setup.py execution