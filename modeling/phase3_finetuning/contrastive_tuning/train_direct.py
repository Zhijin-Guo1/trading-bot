#!/usr/bin/env python3
"""
Direct Training Script for Contrastive Learning
================================================
Alternative to shell script - runs training directly in Python.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up environment variables for training."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Use HF mirror for model downloads
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("Environment configured:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"  HF_ENDPOINT: {os.environ['HF_ENDPOINT']}")
    print()

def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("❌ No GPU detected! Training will be very slow.")
            return False
    except ImportError:
        print("❌ PyTorch not installed properly")
        return False
    return True

def train_with_llamafactory():
    """Run training using LlamaFactory CLI."""
    
    # Training arguments
    args = [
        'llamafactory-cli', 'train',
        '--stage', 'sft',
        '--do_train', 'true',
        '--model_name_or_path', 'Qwen/Qwen2.5-7B-Instruct',
        '--dataset', 'contrastive_8k',
        '--dataset_dir', './data',
        '--template', 'qwen',
        '--finetuning_type', 'lora',
        '--output_dir', './output_contrastive',
        '--overwrite_cache', 'true',
        '--overwrite_output_dir', 'true',
        '--cutoff_len', '2048',
        '--preprocessing_num_workers', '4',
        '--per_device_train_batch_size', '1',
        '--per_device_eval_batch_size', '1',
        '--gradient_accumulation_steps', '8',
        '--lr_scheduler_type', 'cosine',
        '--logging_steps', '10',
        '--warmup_ratio', '0.1',
        '--save_steps', '200',
        '--eval_steps', '100',
        '--evaluation_strategy', 'steps',
        '--load_best_model_at_end', 'true',
        '--learning_rate', '1e-4',
        '--num_train_epochs', '3',
        '--max_grad_norm', '1.0',
        '--plot_loss', 'true',
        '--bf16', 'true',
        '--lora_rank', '32',
        '--lora_alpha', '64',
        '--lora_dropout', '0.1',
        '--gradient_checkpointing', 'true',
        '--val_size', '0.1',
        '--flash_attn', 'fa2'
    ]
    
    print("Starting training with arguments:")
    print("  Model: Qwen/Qwen2.5-7B-Instruct")
    print("  LoRA rank: 32")
    print("  Batch size: 1 (gradient accumulation: 8)")
    print("  Epochs: 3")
    print()
    
    try:
        # Run training
        result = subprocess.run(args, check=True, text=True)
        print("\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ llamafactory-cli not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'llamafactory[torch,metrics]'])
        print("Please run the script again after installation.")
        return False

def train_with_alternative_model():
    """Train with smaller model if Qwen fails."""
    print("\nTrying alternative model: Llama-3.2-3B-Instruct")
    
    args = [
        'llamafactory-cli', 'train',
        '--stage', 'sft',
        '--do_train', 'true',
        '--model_name_or_path', 'meta-llama/Llama-3.2-3B-Instruct',  # Smaller model
        '--dataset', 'contrastive_8k',
        '--dataset_dir', './data',
        '--template', 'llama3',  # Changed template
        '--finetuning_type', 'lora',
        '--output_dir', './output_contrastive_llama',
        '--overwrite_cache', 'true',
        '--overwrite_output_dir', 'true',
        '--cutoff_len', '1024',  # Reduced for smaller model
        '--preprocessing_num_workers', '4',
        '--per_device_train_batch_size', '2',  # Can use larger batch
        '--per_device_eval_batch_size', '2',
        '--gradient_accumulation_steps', '4',
        '--lr_scheduler_type', 'cosine',
        '--logging_steps', '10',
        '--warmup_ratio', '0.1',
        '--save_steps', '200',
        '--eval_steps', '100',
        '--evaluation_strategy', 'steps',
        '--load_best_model_at_end', 'true',
        '--learning_rate', '2e-4',
        '--num_train_epochs', '3',
        '--max_grad_norm', '1.0',
        '--plot_loss', 'true',
        '--bf16', 'true',
        '--lora_rank', '16',  # Smaller rank for smaller model
        '--lora_alpha', '32',
        '--lora_dropout', '0.1',
        '--gradient_checkpointing', 'true',
        '--val_size', '0.1'
    ]
    
    try:
        result = subprocess.run(args, check=True, text=True)
        print("\n✅ Training with Llama model completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Alternative model also failed: {e}")
        return False

def main():
    """Main training function."""
    print("="*60)
    print("CONTRASTIVE LEARNING TRAINING SCRIPT")
    print("="*60)
    
    # Setup
    setup_environment()
    
    # Check GPU
    if not check_gpu():
        print("Please fix GPU issues before continuing.")
        sys.exit(1)
    
    # Check data exists
    if not Path("data/train.json").exists():
        print("❌ Training data not found!")
        print("Please run prepare_contrastive_data.py first.")
        sys.exit(1)
    
    print(f"✅ Found training data: {Path('data/train.json').stat().st_size / 1e6:.1f} MB")
    print()
    
    # Try training with Qwen first
    success = train_with_llamafactory()
    
    # If failed, try alternative
    if not success:
        print("\nWould you like to try with a smaller model? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            success = train_with_alternative_model()
    
    if success:
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check the model in ./output_contrastive/")
        print("2. Run evaluation: python inference_contrastive.py")
        print("3. Test single filing: python inference_single_filing.py")
    else:
        print("\nTroubleshooting tips:")
        print("1. Check GPU memory: nvidia-smi")
        print("2. Try smaller batch size or model")
        print("3. Check error logs in output_contrastive/")

if __name__ == "__main__":
    main()