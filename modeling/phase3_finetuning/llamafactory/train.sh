#!/bin/bash
# Main training script for LlamaFactory with Qwen2.5-1.5B-Instruct

echo "========================================"
echo "Qwen2.5-1.5B-Instruct Fine-tuning"
echo "========================================"

# Kill any existing training
echo "Stopping any existing training..."
pkill -f llamafactory-cli 2>/dev/null
sleep 3

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

# Clear GPU memory thoroughly
echo "Clearing GPU memory..."
python3 -c "
import torch
import gc
gc.collect()
torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
"

# Prepare data
echo -e "\nPreparing data..."
python3 fix_data_format.py
if [ $? -ne 0 ]; then
    echo "Data preparation failed!"
    exit 1
fi

# Start training with memory-optimized config
echo -e "\nStarting memory-optimized training..."
echo "Using config: config_memory_optimized.yaml"
echo "Optimizations: batch_size=2, lora_rank=32, seq_len=1024"

llamafactory-cli train config_memory_optimized.yaml 2>&1 | tee training.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "\n✅ Training completed!"
    echo "Model saved to: ./outputs/qwen2.5-1.5b-lora"
else
    echo -e "\n❌ Training failed - check training.log"
    echo -e "\nIf OOM error, further reduce settings in config_memory_optimized.yaml:"
    echo "  - per_device_train_batch_size: 1"
    echo "  - cutoff_len: 512"
    echo "  - lora_rank: 16"
fi