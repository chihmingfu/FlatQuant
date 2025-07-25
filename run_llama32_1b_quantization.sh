#!/bin/bash

# LLaMA-3.2-1B W4A4KV4 Quantization Script
# This script runs the quantization process without timeout
# Status will be monitored every 10 minutes

echo "Starting LLaMA-3.2-1B W4A4KV4 quantization at $(date)"
echo "Model path: ./modelzoo/llama-3.2-1b"
echo "Output directory: ./outputs"
echo "Configuration: W4A4KV4 with asymmetric KV-cache"

python ./main.py \
    --model ./modelzoo/llama-3.2-1b \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs --save_matrix \
    --lm_eval --lm_eval_batch_size 16 \
    2>&1 | tee llama32_1b_quantization_$(date +%Y%m%d_%H%M%S).log

echo "Quantization completed at $(date)"