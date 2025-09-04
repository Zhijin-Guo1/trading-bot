#!/usr/bin/env python3
"""
Full Binary Classification Evaluation - FIXED for list metadata format
Evaluates model on test set for outperform/underperform predictions
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BinaryEvaluator:
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize evaluator for binary classification."""
        logger.info("="*60)
        logger.info("BINARY CLASSIFICATION EVALUATOR")
        logger.info("="*60)
        logger.info(f"Model path: {model_path}")
        logger.info(f"Base model: {base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        logger.info("✅ Model loaded successfully!\n")
    
    def extract_binary_prediction(self, response: str) -> Tuple[str, float]:
        """Extract POSITIVE/NEGATIVE prediction and confidence from response."""
        response_upper = response.upper()
        
        # Look for explicit POSITIVE/NEGATIVE
        prediction = None
        if "POSITIVE" in response_upper:
            prediction = "POSITIVE"
        elif "NEGATIVE" in response_upper:
            prediction = "NEGATIVE"
        
        # If not found, look for alternatives
        if prediction is None:
            if "OUTPERFORM" in response_upper or "BEAT" in response_upper:
                prediction = "POSITIVE"
            elif "UNDERPERFORM" in response_upper or "MISS" in response_upper:
                prediction = "NEGATIVE"
            elif "FILING A" in response and "BETTER" in response_upper:
                prediction = "A_BETTER"
            elif "FILING B" in response and "BETTER" in response_upper:
                prediction = "B_BETTER"
        
        # Extract confidence if mentioned
        confidence = 0.5  # default
        confidence_patterns = [
            r'confidence:?\s*([\d.]+)',
            r'(\d+)%\s*confident',
            r'confidence\s*=\s*([\d.]+)',
            r'highly confident',  # maps to 0.8
            r'very confident',    # maps to 0.8
            r'somewhat confident', # maps to 0.6
            r'low confidence'     # maps to 0.4
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response.lower())
            if match:
                if 'highly' in pattern or 'very' in pattern:
                    confidence = 0.8
                elif 'somewhat' in pattern:
                    confidence = 0.6
                elif 'low' in pattern:
                    confidence = 0.4
                else:
                    try:
                        conf_val = float(match.group(1))
                        confidence = conf_val if conf_val <= 1 else conf_val/100
                    except:
                        pass
                break
        
        return prediction, confidence
    
    def generate_prediction(self, instruction: str, max_new_tokens: int = 150) -> Tuple[str, float, str]:
        """Generate binary prediction for instruction."""
        inputs = self.tokenizer(
            instruction, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1500
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        prediction, confidence = self.extract_binary_prediction(response)
        return prediction, confidence, response
    
    def evaluate_test_set(self, test_path: str = "data/test.json", 
                         metadata_path: str = "data/test_metadata.json",
                         quick: bool = False):
        """Evaluate on test set."""
        
        # Load test data
        with open(test_path) as f:
            test_data = json.load(f)
        
        with open(metadata_path) as f:
            metadata_list = json.load(f)  # This is a list, not a dict
        
        if quick:
            logger.info("Running QUICK evaluation (20 samples)...")
            test_data = test_data[:20]
            metadata_list = metadata_list[:20] if len(metadata_list) >= 20 else metadata_list
        else:
            logger.info(f"Running FULL evaluation ({len(test_data)} samples)...")
        
        results = []
        pair_results = []
        single_results = []
        
        for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
            instruction = item['instruction']
            expected_output = item.get('output', '')
            
            # Generate prediction
            pred, conf, response = self.generate_prediction(instruction)
            
            # Get corresponding metadata
            meta = metadata_list[i] if i < len(metadata_list) else {}
            
            # Determine if this is a pair comparison or single filing
            is_pair = "Filing A" in instruction and "Filing B" in instruction
            
            if is_pair:
                # For pairs, check which filing is predicted as better
                correct_answer = meta.get('correct_answer', 'Unknown')
                filing_a_return = meta.get('filing_a_return', 0)
                filing_b_return = meta.get('filing_b_return', 0)
                
                # Determine what model predicted
                if "Filing A" in response and ("better" in response.lower() or "outperform" in response.lower()):
                    pair_pred = "A"
                elif "Filing B" in response and ("better" in response.lower() or "outperform" in response.lower()):
                    pair_pred = "B"
                elif pred == "A_BETTER":
                    pair_pred = "A"
                elif pred == "B_BETTER":
                    pair_pred = "B"
                else:
                    # Try to parse from response
                    if "A" in response[:50]:
                        pair_pred = "A"
                    else:
                        pair_pred = "B"
                
                pair_results.append({
                    'index': i,
                    'prediction': pair_pred,
                    'confidence': conf,
                    'actual': correct_answer,
                    'correct': pair_pred == correct_answer,
                    'filing_a_return': filing_a_return,
                    'filing_b_return': filing_b_return,
                    'return_diff': abs(filing_a_return - filing_b_return),
                    'response_preview': response[:200]
                })
            else:
                # Single filing prediction
                actual_return = meta.get('actual_return', 0)
                actual_label = "POSITIVE" if actual_return > 0 else "NEGATIVE"
                
                single_results.append({
                    'index': i,
                    'prediction': pred if pred else "NONE",
                    'confidence': conf,
                    'actual': actual_label,
                    'actual_return': actual_return,
                    'correct': pred == actual_label if pred else False,
                    'response_preview': response[:200]
                })
        
        # Calculate metrics
        self.calculate_metrics(pair_results, single_results, quick)
        
        # Save results
        if pair_results:
            pd.DataFrame(pair_results).to_csv('binary_pairs_results.csv', index=False)
            logger.info(f"Saved {len(pair_results)} pair results to binary_pairs_results.csv")
        
        if single_results:
            pd.DataFrame(single_results).to_csv('binary_singles_results.csv', index=False)
            logger.info(f"Saved {len(single_results)} single results to binary_singles_results.csv")
    
    def calculate_metrics(self, pair_results, single_results, quick=False):
        """Calculate and display evaluation metrics."""
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Pair comparison metrics
        if pair_results:
            df_pairs = pd.DataFrame(pair_results)
            
            print("\n📊 PAIR COMPARISON METRICS")
            print("-" * 40)
            
            # Overall accuracy
            valid_pairs = df_pairs[df_pairs['actual'] != 'Unknown']
            if len(valid_pairs) > 0:
                accuracy = valid_pairs['correct'].mean()
                print(f"Overall Accuracy: {accuracy:.1%} ({len(valid_pairs)} valid pairs)")
                
                # High confidence accuracy
                high_conf = valid_pairs[valid_pairs['confidence'] >= 0.7]
                if len(high_conf) > 0:
                    high_conf_acc = high_conf['correct'].mean()
                    print(f"High Confidence (≥0.7) Accuracy: {high_conf_acc:.1%} ({len(high_conf)} samples)")
                
                # Trading simulation
                print("\n💹 TRADING SIMULATION (Pair Trading)")
                print("-" * 40)
                
                # Only trade high confidence pairs
                trades = valid_pairs[valid_pairs['confidence'] >= 0.6].copy()
                if len(trades) > 0:
                    # Calculate PnL for pair trading
                    trades['pnl'] = trades.apply(
                        lambda x: (x['filing_a_return'] - x['filing_b_return']) if x['prediction'] == 'A' 
                                 else (x['filing_b_return'] - x['filing_a_return']), 
                        axis=1
                    )
                    
                    total_return = trades['pnl'].sum()
                    avg_return = trades['pnl'].mean()
                    win_rate = (trades['pnl'] > 0).mean()
                    sharpe = avg_return / trades['pnl'].std() if trades['pnl'].std() > 0 else 0
                    
                    print(f"Number of Trades: {len(trades)}")
                    print(f"Win Rate: {win_rate:.1%}")
                    print(f"Average Return per Trade: {avg_return:.2%}")
                    print(f"Total Return: {total_return:.1%}")
                    print(f"Sharpe Ratio: {sharpe:.2f}")
            else:
                print("No valid pairs with ground truth found")
        
        # Single filing metrics
        if single_results:
            df_singles = pd.DataFrame(single_results)
            valid_singles = df_singles[df_singles['actual'] != 'Unknown']
            
            if len(valid_singles) > 0:
                print("\n📊 SINGLE FILING METRICS")
                print("-" * 40)
                
                # Binary accuracy
                accuracy = valid_singles['correct'].mean()
                print(f"Binary Classification Accuracy: {accuracy:.1%} ({len(valid_singles)} valid samples)")
                
                # Confusion matrix
                print("\nConfusion Matrix:")
                for actual in ['POSITIVE', 'NEGATIVE']:
                    for pred in ['POSITIVE', 'NEGATIVE']:
                        count = len(valid_singles[(valid_singles['actual'] == actual) & 
                                              (valid_singles['prediction'] == pred)])
                        print(f"  Actual {actual}, Predicted {pred}: {count}")
                
                # Precision and Recall
                tp = len(valid_singles[(valid_singles['actual'] == 'POSITIVE') & 
                                   (valid_singles['prediction'] == 'POSITIVE')])
                fp = len(valid_singles[(valid_singles['actual'] == 'NEGATIVE') & 
                                   (valid_singles['prediction'] == 'POSITIVE')])
                fn = len(valid_singles[(valid_singles['actual'] == 'POSITIVE') & 
                                   (valid_singles['prediction'] == 'NEGATIVE')])
                tn = len(valid_singles[(valid_singles['actual'] == 'NEGATIVE') & 
                                   (valid_singles['prediction'] == 'NEGATIVE')])
                
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                    print(f"\nPrecision (POSITIVE): {precision:.1%}")
                
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    print(f"Recall (POSITIVE): {recall:.1%}")
                
                if tn + fn > 0:
                    specificity = tn / (tn + fn)
                    print(f"Specificity (NEGATIVE): {specificity:.1%}")
                
                # Trading simulation
                print("\n💹 TRADING SIMULATION (Long/Short)")
                print("-" * 40)
                
                # Trade based on predictions
                trades = valid_singles[valid_singles['prediction'].isin(['POSITIVE', 'NEGATIVE'])].copy()
                if len(trades) > 0:
                    trades['position'] = trades['prediction'].map({'POSITIVE': 1, 'NEGATIVE': -1})
                    trades['pnl'] = trades['position'] * trades['actual_return']
                    
                    total_return = trades['pnl'].sum()
                    avg_return = trades['pnl'].mean()
                    win_rate = (trades['pnl'] > 0).mean()
                    sharpe = avg_return / trades['pnl'].std() if trades['pnl'].std() > 0 else 0
                    
                    print(f"Number of Trades: {len(trades)}")
                    print(f"Win Rate: {win_rate:.1%}")
                    print(f"Average Return per Trade: {avg_return:.2%}")
                    print(f"Total Return: {total_return:.1%}")
                    print(f"Sharpe Ratio: {sharpe:.2f}")
            else:
                print("\n📊 SINGLE FILING METRICS")
                print("No valid single filings with ground truth found")
        
        print("\n" + "="*60)
        if quick:
            print("Quick evaluation complete. Run without --quick for full results.")
        else:
            print("Full evaluation complete.")

def main():
    parser = argparse.ArgumentParser(description='Binary Classification Evaluation')
    parser.add_argument('--model_path', default='./output_contrastive',
                       help='Path to fine-tuned model')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation with 20 samples')
    parser.add_argument('--test_path', default='data/test.json',
                       help='Path to test data')
    parser.add_argument('--metadata_path', default='data/test_metadata.json',
                       help='Path to test metadata')
    
    args = parser.parse_args()
    
    evaluator = BinaryEvaluator(args.model_path)
    evaluator.evaluate_test_set(
        test_path=args.test_path,
        metadata_path=args.metadata_path,
        quick=args.quick
    )

if __name__ == "__main__":
    main()