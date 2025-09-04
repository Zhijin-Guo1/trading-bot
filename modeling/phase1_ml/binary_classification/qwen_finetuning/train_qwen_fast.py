#!/usr/bin/env python3
"""
ULTRA-FAST Qwen fine-tuning - Optimized for <1 hour training
Key optimizations:
1. Smallest model (0.5B)
2. Minimal data sampling
3. Reduced sequence length
4. Single epoch
5. Larger batch size
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
import time

def load_data(data_dir="../../../data", sample_size=1000):
    """
    Load minimal data for ultra-fast training
    
    Args:
        data_dir: Directory containing the data
        sample_size: Number of samples to use (1000 = ~10 min training)
    """
    print("Loading data...")
    
    # Load CSVs
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    
    print(f"Original size - Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Ultra-aggressive sampling for speed
    train_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
    val_df = val_df.sample(min(sample_size//5, len(val_df)), random_state=42)  # Even smaller validation
    
    print(f"After sampling - Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Create simple binary labels
    train_df['label'] = (train_df['adjusted_return_pct'] > 0).astype(int)
    val_df['label'] = (val_df['adjusted_return_pct'] > 0).astype(int)
    
    return train_df, val_df

def create_ultra_short_prompt(row):
    """Create minimal prompt for fastest processing"""
    # Only use first 200 chars of text
    text = str(row.get('summary', ''))[:200]
    
    # Minimal prompt format
    prompt = f"Text: {text}\nPredict: "
    return prompt

def prepare_dataset_ultra_fast(df, tokenizer):
    """Prepare dataset with minimal processing"""
    texts = []
    for _, row in df.iterrows():
        prompt = create_ultra_short_prompt(row)
        label = "UP" if row['label'] == 1 else "DOWN"
        texts.append(f"{prompt}{label}")
    
    # Tokenize with very short sequences
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=128,  # Ultra-short sequences
            return_tensors=None
        )
    
    dataset = HFDataset.from_dict({'text': texts})
    tokenized = dataset.map(tokenize_function, batched=True, num_proc=4)
    
    return tokenized

def main():
    print("="*70)
    print("ULTRA-FAST QWEN TRAINING (<1 HOUR)")
    print("="*70)
    
    # ============ AGGRESSIVE OPTIMIZATIONS ============
    
    # Use SMALLEST possible model
    MODEL_OPTIONS = {
        'ultra_fast': "Qwen/Qwen2.5-0.5B-Instruct",  # 500M params - fastest
        'fast': "Qwen/Qwen2.5-1.5B-Instruct",        # 1.5B params - balanced
        'normal': "Qwen/Qwen2.5-3B-Instruct",        # 3B params - better quality
    }
    
    # SELECT YOUR SPEED
    SPEED_MODE = 'ultra_fast'  # Change this to 'fast' or 'normal' if needed
    MODEL_NAME = MODEL_OPTIONS[SPEED_MODE]
    
    # Other optimizations
    SAMPLE_SIZE = 1000      # Use only 1000 samples (vs 14,449)
    MAX_LENGTH = 128        # Very short sequences (vs 512)
    BATCH_SIZE = 16         # Larger batch for faster training
    EPOCHS = 1              # Single epoch only
    LORA_R = 4              # Minimal LoRA rank (vs 8-16)
    
    OUTPUT_DIR = f"./qwen_{SPEED_MODE}_output"
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME} ({SPEED_MODE} mode)")
    print(f"  Samples: {SAMPLE_SIZE}")
    print(f"  Max length: {MAX_LENGTH} tokens")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  LoRA rank: {LORA_R}")
    print(f"\nEstimated time: <1 hour")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected! This script requires GPU.")
        print("CPU training would take 10+ hours even with optimizations.")
        return
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    start_time = time.time()
    
    # Load minimal data
    train_df, val_df = load_data(sample_size=SAMPLE_SIZE)
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load smallest model with 4-bit quantization
    print(f"Loading {SPEED_MODE} model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Minimal LoRA config
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        target_modules=["q_proj", "v_proj"],  # Only essential modules
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_dataset_ultra_fast(train_df, tokenizer)
    val_dataset = prepare_dataset_ultra_fast(val_df, tokenizer)
    
    # Training arguments optimized for speed
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=1,  # No accumulation for speed
        warmup_steps=10,  # Minimal warmup
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=500,
        save_total_limit=1,
        fp16=True,
        dataloader_num_workers=4,
        report_to="none",
        gradient_checkpointing=False,  # Disabled for speed
        optim="adamw_torch",  # Faster optimizer
        tf32=True if torch.cuda.is_available() else False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    # Train
    print("\n" + "="*70)
    print("Starting ULTRA-FAST training...")
    print("="*70 + "\n")
    
    trainer.train()
    
    # Save model
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    
    # Calculate time
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Model saved to: {OUTPUT_DIR}/final_model")
    
    # Quick evaluation
    print("\n🔍 Quick Evaluation:")
    model.eval()
    
    # Test on a few examples
    test_prompts = [
        "Text: Company reports record earnings beating analyst expectations\nPredict: ",
        "Text: Company announces bankruptcy filing and restructuring plan\nPredict: ",
        "Text: Quarterly revenue meets expectations with stable outlook\nPredict: "
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, temperature=0.1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = response.split("Predict: ")[-1].strip()[:10]
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Prediction: {prediction}")
    
    print("\n📊 Training Statistics:")
    print(f"  Model size: {MODEL_NAME.split('-')[-2]}")
    print(f"  Training samples: {SAMPLE_SIZE}")
    print(f"  Sequence length: {MAX_LENGTH} tokens")
    print(f"  Time per sample: {training_time/SAMPLE_SIZE:.2f} seconds")
    
    if training_time > 3600:
        print(f"\n⚠️ Training took {training_time/3600:.1f} hours")
        print("To make it faster:")
        print("1. Use 'ultra_fast' mode (0.5B model)")
        print("2. Reduce SAMPLE_SIZE to 500")
        print("3. Reduce MAX_LENGTH to 64")
        print("4. Increase BATCH_SIZE to 32 (if GPU memory allows)")
    else:
        print(f"\n✨ Success! Training completed in under 1 hour!")

if __name__ == "__main__":
    main()