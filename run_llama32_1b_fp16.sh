#!/bin/bash

# LLaMA-3.2-1B FP16 Baseline Script
echo "Starting LLaMA-3.2-1B FP16 baseline evaluation at $(date)"
echo "Model path: ./modelzoo/llama-3.2-1b"
echo "Configuration: FP16 (16-bit weights, activations, KV-cache)"

python ./main.py \
    --model ./modelzoo/llama-3.2-1b \
    --w_bits 16 --a_bits 16 \
    --k_bits 16 --v_bits 16 \
    --cali_bsz 4 --epoch 0 \
    --output_dir ./outputs \
    --exp_name fp16_baseline_1b \
    --lm_eval --lm_eval_batch_size 16 \
    2>&1 | tee llama32_1b_fp16_$(date +%Y%m%d_%H%M%S).log

echo "FP16 baseline evaluation completed at $(date)"