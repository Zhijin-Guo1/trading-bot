#!/bin/bash
# Simplified training script - minimal parameters to avoid conflicts

# Set environment
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "SIMPLIFIED CONTRASTIVE TRAINING"
echo "=========================================="
echo "Using minimal parameters to avoid conflicts"
echo ""

# Run with minimal required parameters
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset contrastive_8k \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --output_dir ./output_contrastive \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --save_steps 500 \
    --bf16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_checkpointing

echo "Training complete!"