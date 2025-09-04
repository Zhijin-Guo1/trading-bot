#!/bin/bash
# Training script for contrastive learning model
# Run this on your RTX 3090 GPU server

# Check if LlamaFactory is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "Installing LlamaFactory..."
    pip install llamafactory[torch,metrics]
fi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# Use Hugging Face mirror for China/restricted regions
export HF_ENDPOINT=https://hf-mirror.com

echo "=========================================="
echo "Starting Contrastive Learning Fine-tuning"
echo "=========================================="
echo "Model: Qwen2.5-7B-Instruct"
echo "Task: 8-K Filing Comparison"
echo "GPU: RTX 3090 (24GB)"
echo "Using HF Mirror: $HF_ENDPOINT"
echo ""

# Run training with LlamaFactory - pass all parameters explicitly
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset contrastive_8k \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --output_dir ./output_contrastive \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --max_grad_norm 1.0 \
    --plot_loss \
    --bf16 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --gradient_checkpointing \
    --val_size 0.1

echo ""
echo "Training complete! Model saved to ./output_contrastive"
echo ""
echo "To run inference, use:"
echo "python inference_contrastive.py --model_path ./output_contrastive"