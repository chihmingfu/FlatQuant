#!/bin/bash

# Run LLaMA-3.2-3B W4A4KV4 perplexity evaluation
python ./main.py \
    --model ./modelzoo/llama-3/llama-3.2-3b \
    --w_bits 4 --a_bits 4 \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \
    --lwc --lac --cali_trans --add_diag \
    --output_dir ./outputs \
    --reload_matrix --matrix_path ./outputs/llama-3.2-3b/w4a4/exp \
    --lm_eval --lm_eval_batch_size 16 \
    2>&1 | tee llama32_ppl_$(date +%Y%m%d_%H%M%S).log