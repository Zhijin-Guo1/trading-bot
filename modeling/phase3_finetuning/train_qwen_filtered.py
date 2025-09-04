#!/usr/bin/env python3
"""
Fine-tune Qwen-2.5-7B for Stock Movement Prediction using Filtered 8-K Data
Optimized for GPU training with QLoRA (4-bit quantization)
Uses cleaned data with boilerplate removed
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Transformers and PEFT imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset as HFDataset
import evaluate

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class StockPredictionDataset:
    """Dataset for stock prediction with improved instruction format"""
    
    def __init__(self, data_path: str, text_column: str = 'summary'):
        """
        Initialize dataset from filtered CSV
        
        Args:
            data_path: Path to filtered CSV file
            text_column: Column containing the text (summary or full_text)
        """
        self.data = pd.read_csv(data_path)
        self.text_column = text_column
        
        # Check available columns
        print(f"  Loaded {len(self.data)} samples")
        print(f"  Columns: {self.data.columns.tolist()[:10]}...")
        
        # Create binary labels from returns
        if 'adjusted_return_pct' in self.data.columns:
            self.data['binary_label'] = (self.data['adjusted_return_pct'] > 0).astype(int)
            self.data['label_text'] = self.data['binary_label'].map({1: 'UP', 0: 'DOWN'})
        else:
            raise ValueError("No return column found in data")
            
        # Check text column
        if text_column not in self.data.columns:
            print(f"  Warning: {text_column} not found, using first text-like column")
            text_cols = [c for c in self.data.columns if 'text' in c.lower() or 'summary' in c.lower()]
            if text_cols:
                self.text_column = text_cols[0]
                print(f"  Using column: {self.text_column}")
            else:
                raise ValueError(f"No text column found in data")
        
        # Add immediate reaction if available
        if 'immediate_reaction_pct' in self.data.columns:
            self.has_immediate = True
        else:
            self.has_immediate = False
            print("  Note: No immediate reaction data available")
    
    def create_prompt(self, row: pd.Series) -> str:
        """Create improved instruction prompt for the model"""
        
        # Build context information
        context_parts = []
        
        # Basic info
        context_parts.append(f"Ticker: {row.get('ticker', 'N/A')}")
        context_parts.append(f"Filing Date: {row.get('filing_date', 'N/A')}")
        
        # Sector/Industry if available
        if 'sector' in row:
            context_parts.append(f"Sector: {row.get('sector', 'N/A')}")
        if 'industry' in row:
            context_parts.append(f"Industry: {row.get('industry', 'N/A')}")
        
        # 8-K items if available
        if 'items_present' in row:
            context_parts.append(f"8-K Items: {row.get('items_present', 'N/A')}")
        
        # Market momentum if available
        momentum_parts = []
        if 'momentum_7d' in row:
            momentum_parts.append(f"7-day: {row.get('momentum_7d', 0):.2f}%")
        if 'momentum_30d' in row:
            momentum_parts.append(f"30-day: {row.get('momentum_30d', 0):.2f}%")
        
        if momentum_parts:
            context_parts.append(f"Recent Momentum: {', '.join(momentum_parts)}")
        
        # Immediate reaction (for analysis, not training feature)
        if self.has_immediate and 'immediate_reaction_pct' in row:
            # Don't include in training to avoid leakage
            pass
        
        context = '\n'.join(context_parts)
        
        # Get text content
        text_content = str(row.get(self.text_column, 'No content available'))
        
        # Truncate if too long (keep under token limit)
        max_text_length = 2000  # characters, roughly 500 tokens
        if len(text_content) > max_text_length:
            text_content = text_content[:max_text_length] + "..."
        
        # Create the instruction prompt
        instruction = """You are a financial analyst predicting stock movements based on 8-K SEC filings.
Task: Analyze the following 8-K filing to predict if the stock price will increase or decrease over the next 5 trading days.

IMPORTANT: Respond with exactly one word: either "UP" or "DOWN"."""

        full_prompt = f"""{instruction}

Context:
{context}

8-K Filing Content:
{text_content}

Prediction:"""
        
        return full_prompt
    
    def create_training_example(self, idx: int) -> Dict:
        """Create a training example with prompt and response"""
        row = self.data.iloc[idx]
        prompt = self.create_prompt(row)
        label = row['label_text']
        
        # Format as instruction-response pair
        full_text = f"{prompt} {label}"
        
        return {
            'text': full_text,
            'prompt': prompt,
            'response': label,
            'label': label,
            'binary_label': row['binary_label']
        }
    
    def prepare_dataset(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Prepare all examples for training"""
        examples = []
        
        n_samples = min(max_samples, len(self.data)) if max_samples else len(self.data)
        
        for idx in range(n_samples):
            examples.append(self.create_training_example(idx))
        
        # Print label distribution
        labels = [ex['label'] for ex in examples]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Label distribution: {dict(zip(unique, counts))}")
        
        return examples

def load_and_prepare_data(
    data_dir: str, 
    text_column: str = 'summary',
    max_samples: Optional[int] = None
):
    """Load and prepare filtered datasets for training"""
    print("\nLoading filtered datasets...")
    
    # Use filtered data
    train_path = os.path.join(data_dir, 'filtered_train.csv')
    val_path = os.path.join(data_dir, 'filtered_val.csv')
    test_path = os.path.join(data_dir, 'filtered_test.csv')
    
    # Check if filtered data exists
    if not os.path.exists(train_path):
        print("  Filtered data not found, using original data")
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'val.csv')
        test_path = os.path.join(data_dir, 'test.csv')
    
    # Load datasets
    print("\nTrain set:")
    train_dataset = StockPredictionDataset(train_path, text_column)
    train_examples = train_dataset.prepare_dataset(max_samples)
    
    print("\nValidation set:")
    val_dataset = StockPredictionDataset(val_path, text_column)
    val_examples = val_dataset.prepare_dataset(max_samples // 5 if max_samples else None)
    
    print("\nTest set:")
    test_dataset = StockPredictionDataset(test_path, text_column)
    test_examples = test_dataset.prepare_dataset(max_samples // 5 if max_samples else None)
    
    # Convert to HuggingFace datasets
    train_hf = HFDataset.from_list(train_examples)
    val_hf = HFDataset.from_list(val_examples)
    test_hf = HFDataset.from_list(test_examples)
    
    return train_hf, val_hf, test_hf

def setup_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    load_in_8bit: bool = False
):
    """Setup Qwen model with QLoRA configuration"""
    print(f"\nLoading model: {model_name}")
    
    # Quantization configuration
    if load_in_8bit:
        # 8-bit quantization (more accurate, uses more memory)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
    else:
        # 4-bit quantization (more efficient)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration (optimized for stock prediction)
    lora_config = LoraConfig(
        r=32,  # Increased rank for better capacity
        lora_alpha=64,  # Alpha = 2 * r
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,  # Reduced dropout
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize examples for training"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    
    # For generation tasks, predictions are logits
    # We need to extract the actual predictions
    if len(predictions.shape) == 3:
        # Take argmax to get token IDs
        predictions = np.argmax(predictions, axis=-1)
    
    # Simple accuracy based on matching sequences
    # This is a simplified version - in practice you'd decode and compare
    accuracy = np.mean(predictions == labels)
    
    return {'accuracy': accuracy}

def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 0.3,
    weight_decay: float = 0.001
):
    """Train the model using HuggingFace Trainer with optimized settings"""
    print("\nStarting training with optimized parameters...")
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Calculate training steps
    total_steps = (len(train_tokenized) // (batch_size * gradient_accumulation_steps)) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Training arguments with optimizations
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        fp16=True,  # Mixed precision training
        bf16=False,  # Use fp16 instead of bf16 for better compatibility
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # Use loss instead of accuracy
        greater_is_better=False,
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=True,  # Save memory
        optim="adamw_torch",  # Optimizer
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    
    return trainer

def evaluate_model(model, tokenizer, test_dataset, batch_size=2):
    """Evaluate model on test set with batch processing and memory optimization"""
    print("\nEvaluating on test set...")
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    all_probs = []
    
    # Process in batches for efficiency
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        return batch
    
    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx}/{len(dataloader)}")
            
            try:
                for example in batch:
                    # Clear cache before processing each example
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Prepare input with reduced max_length
                    inputs = tokenizer(
                        example['prompt'],
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,  # Reduced from 1024
                        padding=True
                    ).to(model.device)
                    
                    # Generate prediction with minimal tokens
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=3,  # Reduced from 5
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode prediction
                    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract predicted label
                    pred_text_clean = pred_text.replace(example['prompt'], '').strip()
                    
                    # Look for UP or DOWN in the response
                    if 'UP' in pred_text_clean.upper()[:10]:
                        pred_label = 'UP'
                        prob = 0.8  # Placeholder probability
                    elif 'DOWN' in pred_text_clean.upper()[:10]:
                        pred_label = 'DOWN'
                        prob = 0.2  # Placeholder probability
                    else:
                        # Default to DOWN if unclear
                        pred_label = 'DOWN'
                        prob = 0.5
                    
                    all_predictions.append(pred_label)
                    all_true_labels.append(example['label'])
                    all_probs.append(prob)
                    
                    # Clear output tensors from memory
                    del outputs, inputs
                    
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch {batch_idx}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                # Skip this batch if OOM
                continue
    
    # Calculate metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Calculate AUC if we have probabilities
    binary_true = [1 if label == 'UP' else 0 for label in all_true_labels]
    if len(set(binary_true)) > 1:  # Need both classes for AUC
        auc = roc_auc_score(binary_true, all_probs)
        print(f"AUC Score: {auc:.4f}")
    else:
        auc = None
    
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_true_labels, all_predictions, labels=['UP', 'DOWN'])
    print("       Pred")
    print("       UP   DOWN")
    print(f"True UP    {cm[0][0]:4d} {cm[0][1]:4d}")
    print(f"     DOWN  {cm[1][0]:4d} {cm[1][1]:4d}")
    
    return accuracy, auc, all_predictions

def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen-2.5-7B on filtered 8-K data')
    parser.add_argument('--data_dir', type=str, default='modeling/phase1_ml/data',
                        help='Directory containing filtered train/val/test CSVs')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--output_dir', type=str, default='./qwen_finetuned_filtered',
                        help='Output directory for model')
    parser.add_argument('--text_column', type=str, default='summary',
                        help='Column containing text data')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to use (for testing)')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='Use 8-bit quantization instead of 4-bit')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QWEN-2.5-7B FINE-TUNING ON FILTERED 8-K DATA")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Quantization: {'8-bit' if args.load_in_8bit else '4-bit'}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
    else:
        print("\n⚠️ Warning: No GPU detected. Training will be very slow!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        args.data_dir,
        args.text_column,
        args.max_samples
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        args.load_in_8bit
    )
    
    # Train model
    start_time = time.time()
    trainer = train_model(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate on test set
    test_accuracy, test_auc, predictions = evaluate_model(
        model,
        tokenizer,
        test_dataset
    )
    
    # Save results
    results = {
        'model': args.model_name,
        'data_dir': args.data_dir,
        'text_column': args.text_column,
        'test_accuracy': float(test_accuracy),
        'test_auc': float(test_auc) if test_auc else None,
        'training_time_minutes': training_time / 60,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'quantization': '8-bit' if args.load_in_8bit else '4-bit',
        'timestamp': datetime.now().isoformat(),
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        }
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    test_data = pd.read_csv(os.path.join(args.data_dir, 'filtered_test.csv'))
    test_data['predicted_label'] = predictions[:len(test_data)]
    test_data.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)
    
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    if test_auc:
        print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Predictions saved to: {args.output_dir}/test_predictions.csv")

if __name__ == '__main__':
    main()