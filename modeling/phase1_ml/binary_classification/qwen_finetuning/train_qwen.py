#!/usr/bin/env python3
"""
Fine-tune Qwen-2.5-7B for Stock Movement Prediction using QLoRA
Optimized for GPU training with 4-bit quantization
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Transformers and PEFT imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
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
    """Dataset for stock prediction with instruction format"""

    def __init__(self, data_path: str, label_type: str = 'binary'):
        """Initialize dataset"""
        # Support both JSON and CSV files
        if data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        else:
            self.data = pd.read_csv(data_path)
        self.label_type = label_type

        # Print data info for debugging
        print(f"    Loaded {len(self.data)} samples")
        print(f"    Columns: {list(self.data.columns)[:10]}...")

        # Create or convert labels for CSV files
        if 'adjusted_return_pct' in self.data.columns:
            # For binary classification
            if 'binary_label' not in self.data.columns:
                self.data['binary_label'] = (self.data['adjusted_return_pct'] > 0).map({True: 'UP', False: 'DOWN'})
            elif self.data['binary_label'].dtype in [int, float, np.int64, np.float64]:
                # Convert numeric to text labels
                self.data['binary_label'] = self.data['binary_label'].map({1: 'UP', 0: 'DOWN'})
            # If already text labels, keep them as is

            # For three-class classification
            if 'three_class_label' not in self.data.columns:
                conditions = [
                    self.data['adjusted_return_pct'] > 1,
                    self.data['adjusted_return_pct'] < -1
                ]
                choices = ['UP', 'DOWN']
                self.data['three_class_label'] = np.select(conditions, choices, default='STAY')
            elif self.data['three_class_label'].dtype in [int, float, np.int64, np.float64]:
                # Convert numeric to text labels if needed
                self.data['three_class_label'] = self.data['three_class_label'].map({0: 'DOWN', 1: 'STAY', 2: 'UP'})
            # If already text labels, keep them as is

        # Print label distribution
        if self.label_type == 'binary' and 'binary_label' in self.data.columns:
            unique, counts = np.unique(self.data['binary_label'].values, return_counts=True)
            print(f"    Binary label distribution: {dict(zip(unique, counts))}")
        elif self.label_type == 'three_class' and 'three_class_label' in self.data.columns:
            unique, counts = np.unique(self.data['three_class_label'].values, return_counts=True)
            print(f"    Three-class label distribution: {dict(zip(unique, counts))}")

    def create_prompt(self, row: pd.Series) -> str:
        """Create instruction prompt for the model"""

        # Extract key features
        momentum_info = f"""
Recent Performance:
- 7-day momentum: {row.get('momentum_7d', 0):.2f}%
- 30-day momentum: {row.get('momentum_30d', 0):.2f}%
- 90-day momentum: {row.get('momentum_90d', 0):.2f}%"""

        market_info = f"VIX (volatility): {row.get('vix_level', 0):.2f}" if 'vix_level' in row else ""

        # Create the instruction prompt
        if self.label_type == 'binary':
            instruction = """You are a financial analyst predicting stock movements based on 8-K filings.
Analyze the following 8-K filing and market context to predict if the stock will go UP or DOWN in the next trading day.

IMPORTANT: Respond with ONLY one word: either "UP" or "DOWN"."""
        else:
            instruction = """You are a financial analyst predicting stock movements based on 8-K filings.
Analyze the following 8-K filing and market context to predict if the stock will go UP (>1%), DOWN (<-1%), or STAY (-1% to 1%) in the next trading day.

IMPORTANT: Respond with ONLY one word: either "UP", "DOWN", or "STAY"."""

        # Handle filing date - could be string or timestamp
        filing_date_str = 'N/A'
        if 'filing_date' in row:
            filing_date = row.get('filing_date')
            if pd.notna(filing_date):
                try:
                    # Try parsing as timestamp (milliseconds)
                    if isinstance(filing_date, (int, float)):
                        filing_date_str = pd.to_datetime(filing_date, unit='ms').strftime('%Y-%m-%d')
                    else:
                        # Already a string date
                        filing_date_str = str(filing_date)
                except:
                    filing_date_str = str(filing_date)

        context = f"""
Stock: {row.get('ticker', 'N/A')}
Sector: {row.get('sector', 'N/A')}
Industry: {row.get('industry', 'N/A')}
Filing Date: {filing_date_str}
8-K Items: {row.get('items_present', 'N/A')}

{momentum_info}
{market_info}

8-K Filing Summary:
{row.get('summary', 'No summary available')}
"""

        full_prompt = f"{instruction}\n{context}\n\nPrediction:"

        return full_prompt

    def create_training_example(self, row: pd.Series) -> Dict:
        """Create a training example with prompt and response"""
        prompt = self.create_prompt(row)

        # Get the label
        if self.label_type == 'binary':
            label = row['binary_label']
        else:
            label = row['three_class_label']

        # Format as instruction-response pair
        full_text = f"{prompt} {label}"

        return {
            'text': full_text,
            'prompt': prompt,
            'response': label,
            'label': label
        }

    def prepare_dataset(self) -> List[Dict]:
        """Prepare all examples for training"""
        examples = []
        for _, row in self.data.iterrows():
            examples.append(self.create_training_example(row))
        return examples


def load_and_prepare_data(data_dir: str, label_type: str = 'binary', use_filtered: bool = True):
    """Load and prepare datasets for training"""
    print("Loading and preparing datasets...")

    # Determine file paths based on whether to use filtered data
    if use_filtered:
        train_path = os.path.join(data_dir, 'train.csv')
        val_path = os.path.join(data_dir, 'val.csv')
        test_path = os.path.join(data_dir, 'test.csv')

        # Fall back to unfiltered if filtered doesn't exist
        if not os.path.exists(train_path):
            print("  Filtered data not found, using unfiltered CSV files")
            train_path = os.path.join(data_dir, 'train.csv')
            val_path = os.path.join(data_dir, 'val.csv')
            test_path = os.path.join(data_dir, 'test.csv')
        else:
            print("  Using filtered CSV files (boilerplate removed)")
    else:
        # Check for JSON files first, then CSV
        if os.path.exists(os.path.join(data_dir, 'train.json')):
            train_path = os.path.join(data_dir, 'train.json')
            val_path = os.path.join(data_dir, 'val.json')
            test_path = os.path.join(data_dir, 'test.json')
        else:
            train_path = os.path.join(data_dir, 'train.csv')
            val_path = os.path.join(data_dir, 'val.csv')
            test_path = os.path.join(data_dir, 'test.csv')

    print(f"  Loading from: {os.path.basename(train_path)}")

    # Load datasets
    train_dataset = StockPredictionDataset(train_path, label_type)
    val_dataset = StockPredictionDataset(val_path, label_type)
    test_dataset = StockPredictionDataset(test_path, label_type)

    # Prepare examples
    train_examples = train_dataset.prepare_dataset()
    val_examples = val_dataset.prepare_dataset()
    test_examples = test_dataset.prepare_dataset()

    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val: {len(val_examples)} examples")
    print(f"  Test: {len(test_examples)} examples")

    # Convert to HuggingFace datasets
    train_hf = HFDataset.from_list(train_examples)
    val_hf = HFDataset.from_list(val_examples)
    test_hf = HFDataset.from_list(test_examples)

    return train_hf, val_hf, test_hf


def setup_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct", use_smaller_model: bool = False):
    """Setup Qwen model with QLoRA configuration"""

    # Option to use smaller model for testing
    if use_smaller_model:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # Much smaller model
        print(f"\nUsing smaller model for testing: {model_name}")
    else:
        print(f"\nLoading model: {model_name}")

    print("Note: Model download may take 10-20 minutes on first run...")

    # QLoRA configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load tokenizer first (smaller download)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="./model_cache",  # Use local cache
        local_files_only=False  # Allow downloading
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Tokenizer loaded successfully!")

    # Load model with quantization
    print("Loading model (this may take a while)...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="./model_cache",  # Use local cache
                local_files_only=False,  # Allow downloading
                resume_download=True,  # Resume if interrupted
                low_cpu_mem_usage=True  # Reduce memory usage during loading
            )
            print("Model loaded successfully!")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("\nERROR: Model download failed after multiple attempts.")
                print("Possible solutions:")
                print("1. Check your internet connection")
                print("2. Try using the smaller model with --use_smaller_model")
                print("3. Pre-download the model using:")
                print(
                    "   python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', cache_dir='./model_cache')\"")
                raise e

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
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


def compute_metrics(eval_pred, tokenizer, label_type='binary'):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Extract predicted labels
    pred_labels = []
    true_labels = []

    valid_labels = ['UP', 'DOWN'] if label_type == 'binary' else ['UP', 'DOWN', 'STAY']

    for pred, true in zip(decoded_preds, decoded_labels):
        # Extract the last word as prediction
        pred_word = pred.strip().split()[-1].upper() if pred.strip() else 'NONE'
        true_word = true.strip().split()[-1].upper() if true.strip() else 'NONE'

        # Validate and default to first valid label if invalid
        if pred_word not in valid_labels:
            pred_word = valid_labels[0]
        if true_word not in valid_labels:
            true_word = valid_labels[0]

        pred_labels.append(pred_word)
        true_labels.append(true_word)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    return {'accuracy': accuracy}


def train_model(
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir: str,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        eval_steps: int = 200
):
    """Train the model using HuggingFace Trainer"""
    print("\nStarting training...")

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

    # Ensure save_steps is a multiple of eval_steps
    save_steps = eval_steps * 2  # Save every 2 evaluation steps

    # Training arguments (compatible with different transformers versions)
    try:
        # Try with eval_strategy (newer versions)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to="none",
            push_to_hub=False,
        )
    except TypeError:
        # Fallback for older versions or compatibility issues
        print("Note: Using simplified training arguments for compatibility")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,  # Reduced to prevent OOM during evaluation
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=50,
            save_steps=save_steps,
            save_total_limit=2,
            report_to="none",
        )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(os.path.join(output_dir, "final_model"))

    return trainer


def evaluate_model(model, tokenizer, test_dataset, label_type='binary'):
    """Evaluate model on test set with memory optimization"""
    print("\nEvaluating on test set...")

    model.eval()
    predictions = []
    true_labels = []

    valid_labels = ['UP', 'DOWN'] if label_type == 'binary' else ['UP', 'DOWN', 'STAY']

    with torch.no_grad():
        for idx, example in enumerate(test_dataset):
            # Clear cache periodically
            if idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if idx > 0:
                    print(f"  Processed {idx}/{len(test_dataset)} samples...")

            try:
                # Prepare input with reduced max_length
                inputs = tokenizer(
                    example['prompt'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512  # Reduced from 1024
                ).to(model.device)

                # Generate prediction with minimal tokens
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,  # Reduced from 10
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

                # Decode prediction
                pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract predicted label
                pred_words = pred_text.replace(example['prompt'], '').strip().split()
                pred_label = pred_words[0].upper() if pred_words else valid_labels[0]

                if pred_label not in valid_labels:
                    pred_label = valid_labels[0]

                predictions.append(pred_label)
                true_labels.append(example['label'].upper())

                # Clear tensors from memory
                del outputs, inputs

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at sample {idx}, clearing cache and continuing...")
                torch.cuda.empty_cache()
                # Use default prediction if OOM
                predictions.append(valid_labels[0])
                true_labels.append(example['label'].upper())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions, labels=valid_labels)
    for i, label in enumerate(valid_labels):
        print(f"  {label:5s}: {cm[i]}")

    return accuracy, predictions


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen-2.5-7B for stock prediction')
    parser.add_argument('--data_dir', type=str, default='../../data',
                        help='Directory containing train/val/test data')
    parser.add_argument('--use_filtered', action='store_true', default=True,
                        help='Use filtered CSV files (default: True)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--output_dir', type=str, default='./qwen_finetuned',
                        help='Output directory for model')
    parser.add_argument('--label_type', type=str, default='binary',
                        choices=['binary', 'three_class'],
                        help='Classification type')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to use (for testing)')
    parser.add_argument('--use_smaller_model', action='store_true',
                        help='Use Qwen2.5-1.5B instead of 7B for testing')

    args = parser.parse_args()

    print("=" * 60)
    print("QWEN-2.5-7B FINE-TUNING FOR STOCK PREDICTION")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWarning: No GPU detected. This will be very slow!")

    # Load and prepare data
    train_dataset, val_dataset, test_dataset = load_and_prepare_data(
        args.data_dir,
        args.label_type,
        use_filtered=args.use_filtered
    )

    # Limit samples if specified (for testing)
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(args.max_samples // 5, len(val_dataset))))
        test_dataset = test_dataset.select(range(min(args.max_samples // 5, len(test_dataset))))
        print(f"\nLimited to {args.max_samples} training samples for testing")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.use_smaller_model)

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

    print(f"\nTraining completed in {training_time / 60:.1f} minutes")

    # Evaluate on test set
    test_accuracy, predictions = evaluate_model(
        model,
        tokenizer,
        test_dataset,
        args.label_type
    )

    # Save results
    results = {
        'model': args.model_name,
        'label_type': args.label_type,
        'test_accuracy': float(test_accuracy),
        'training_time_minutes': training_time / 60,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == '__main__':
    main()