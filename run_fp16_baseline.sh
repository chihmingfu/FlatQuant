#!/bin/bash

# Run FP16 baseline evaluation for LLaMA-3.2-3B
python ./main.py \
    --model ./modelzoo/llama-3/llama-3.2-3b \
    --w_bits 16 --a_bits 16 \
    --k_bits 16 --v_bits 16 \
    --cali_bsz 4 --epoch 0 \
    --output_dir ./outputs \
    --exp_name fp16_baseline \
    --lm_eval --lm_eval_batch_size 16 \
    2>&1 | tee fp16_baseline_$(date +%Y%m%d_%H%M%S).log