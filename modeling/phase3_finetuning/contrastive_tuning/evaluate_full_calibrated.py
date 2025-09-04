#!/usr/bin/env python3
"""
Full Binary Classification Evaluation with THRESHOLD CALIBRATION
Adjusts decision threshold to reduce positive bias
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

class CalibratedBinaryEvaluator:
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct", 
                 confidence_threshold: float = 0.7):
        """Initialize evaluator with calibrated threshold."""
        logger.info("="*60)
        logger.info("CALIBRATED BINARY EVALUATOR")
        logger.info("="*60)
        logger.info(f"Model path: {model_path}")
        logger.info(f"Base model: {base_model}")
        logger.info(f"Confidence threshold for POSITIVE: {confidence_threshold}")
        logger.info("="*60)
        
        self.confidence_threshold = confidence_threshold
        
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
    
    def parse_pair_prediction(self, response: str) -> str:
        """Improved parsing logic for pair comparisons."""
        
        # Take only the first sentence/paragraph to avoid repetition
        first_part = response.split('\n')[0] if '\n' in response else response
        first_100_chars = response[:100]
        
        # Look for explicit statements
        patterns = [
            (r'Filing ([AB]) resulted in better', 1),
            (r'Filing ([AB]) performed better', 1),
            (r'Filing ([AB]) outperformed', 1),
            (r'([AB]) resulted in better', 1),
            (r'([AB]) performed better', 1),
            (r'Filing ([AB]) .*better', 1),
            (r'Answer:\s*Filing ([AB])', 1),
            (r'Answer:\s*([AB])', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, first_part, re.IGNORECASE)
            if match:
                return match.group(group).upper()
        
        # Count occurrences in first few lines
        lines = response.split('\n')[:3]
        text_to_check = ' '.join(lines)
        
        a_count = len(re.findall(r'Filing A', text_to_check, re.IGNORECASE))
        b_count = len(re.findall(r'Filing B', text_to_check, re.IGNORECASE))
        
        if a_count > b_count:
            return "A"
        elif b_count > a_count:
            return "B"
        
        # Default to what appears first
        a_pos = response.lower().find('filing a')
        b_pos = response.lower().find('filing b')
        
        if a_pos != -1 and b_pos != -1:
            return "A" if a_pos < b_pos else "B"
        elif a_pos != -1:
            return "A"
        elif b_pos != -1:
            return "B"
        
        return "UNCLEAR"
    
    def extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from model response."""
        
        response_lower = response.lower()
        
        # Look for explicit confidence mentions
        confidence_patterns = [
            (r'confidence:\s*high', 0.85),
            (r'confidence:\s*medium', 0.6),
            (r'confidence:\s*low', 0.4),
            (r'high confidence', 0.85),
            (r'highly confident', 0.85),
            (r'very confident', 0.8),
            (r'medium confidence', 0.6),
            (r'moderate confidence', 0.6),
            (r'somewhat confident', 0.55),
            (r'low confidence', 0.4),
            (r'confidence:\s*([\d.]+)', 'extract'),
            (r'(\d+)%\s*confident', 'extract_pct'),
        ]
        
        for pattern, value in confidence_patterns:
            match = re.search(pattern, response_lower)
            if match:
                if value == 'extract':
                    try:
                        return float(match.group(1))
                    except:
                        pass
                elif value == 'extract_pct':
                    try:
                        return float(match.group(1)) / 100
                    except:
                        pass
                else:
                    return value
        
        # Look for strong positive/negative language as proxy for confidence
        strong_positive = ['definitely', 'clearly', 'obviously', 'strong', 'significant']
        weak_words = ['might', 'possibly', 'perhaps', 'may', 'could', 'uncertain']
        
        strong_count = sum(1 for word in strong_positive if word in response_lower)
        weak_count = sum(1 for word in weak_words if word in response_lower)
        
        if strong_count > weak_count:
            return 0.75
        elif weak_count > strong_count:
            return 0.45
        
        return 0.6  # Default medium confidence
    
    def calibrate_prediction(self, raw_prediction: str, confidence: float) -> str:
        """Apply threshold calibration to reduce positive bias."""
        
        # For single filing predictions
        if raw_prediction == "POSITIVE":
            # Only keep POSITIVE if confidence is above threshold
            if confidence >= self.confidence_threshold:
                return "POSITIVE"
            else:
                return "NEGATIVE"  # Convert low-confidence positives to negative
        
        return raw_prediction  # Keep NEGATIVE and other predictions as-is
    
    def extract_binary_prediction(self, response: str, is_pair: bool = False) -> Tuple[str, float, str]:
        """Extract prediction with confidence and apply calibration."""
        
        # For pair comparisons
        if is_pair:
            pair_pred = self.parse_pair_prediction(response)
            confidence = self.extract_confidence_score(response)
            return pair_pred, confidence, pair_pred  # No calibration for pairs
        
        # For single predictions
        response_upper = response.upper()
        
        # Extract raw prediction
        raw_prediction = None
        if "POSITIVE" in response_upper:
            raw_prediction = "POSITIVE"
        elif "NEGATIVE" in response_upper:
            raw_prediction = "NEGATIVE"
        
        if raw_prediction is None:
            if "OUTPERFORM" in response_upper or "BEAT" in response_upper:
                raw_prediction = "POSITIVE"
            elif "UNDERPERFORM" in response_upper or "MISS" in response_upper:
                raw_prediction = "NEGATIVE"
        
        # Extract confidence
        confidence = self.extract_confidence_score(response)
        
        # Apply calibration
        if raw_prediction:
            calibrated_prediction = self.calibrate_prediction(raw_prediction, confidence)
        else:
            calibrated_prediction = "NEGATIVE"  # Default to negative if unclear
        
        return calibrated_prediction, confidence, raw_prediction
    
    def generate_prediction(self, instruction: str, is_pair: bool = False) -> Tuple[str, float, str, str]:
        """Generate prediction with calibration."""
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
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        calibrated_pred, confidence, raw_pred = self.extract_binary_prediction(response, is_pair=is_pair)
        return calibrated_pred, confidence, raw_pred, response
    
    def evaluate_test_set(self, test_path: str = "data/test.json", 
                         metadata_path: str = "data/test_metadata.json",
                         quick: bool = False,
                         verbose: bool = False):
        """Evaluate on test set with calibration."""
        
        # Load test data
        with open(test_path) as f:
            test_data = json.load(f)
        
        with open(metadata_path) as f:
            metadata_list = json.load(f)
        
        if quick:
            logger.info("Running QUICK evaluation (first 50 samples)...")
            test_data = test_data[:50]
            metadata_list = metadata_list[:50] if len(metadata_list) >= 50 else metadata_list
        else:
            logger.info(f"Running FULL evaluation ({len(test_data)} samples)...")
        
        pair_results = []
        single_results = []
        calibration_stats = {'changed': 0, 'total': 0}
        
        for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
            instruction = item['instruction']
            
            # Get metadata
            meta = metadata_list[i] if i < len(metadata_list) else {}
            
            # Determine type
            is_pair = "Filing A" in instruction and "Filing B" in instruction
            
            # Generate prediction
            calibrated_pred, confidence, raw_pred, response = self.generate_prediction(instruction, is_pair=is_pair)
            
            if verbose and not is_pair:
                if calibrated_pred != raw_pred:
                    print(f"\nSample {i}: Raw={raw_pred}, Conf={confidence:.2f}, Calibrated={calibrated_pred}")
            
            if is_pair:
                # Pair results
                correct_answer = meta.get('correct_answer', 'Unknown')
                filing_a_return = meta.get('filing_a_return', 0)
                filing_b_return = meta.get('filing_b_return', 0)
                
                pair_results.append({
                    'index': i,
                    'prediction': calibrated_pred,
                    'confidence': confidence,
                    'actual': correct_answer,
                    'correct': calibrated_pred == correct_answer,
                    'filing_a_return': filing_a_return,
                    'filing_b_return': filing_b_return,
                    'return_diff': abs(filing_a_return - filing_b_return),
                    'response_preview': response[:200]
                })
            else:
                # Single filing results
                actual_return = meta.get('actual_return', 0)
                actual_label = "POSITIVE" if actual_return > 0 else "NEGATIVE"
                
                # Track calibration changes
                calibration_stats['total'] += 1
                if calibrated_pred != raw_pred:
                    calibration_stats['changed'] += 1
                
                single_results.append({
                    'index': i,
                    'raw_prediction': raw_pred if raw_pred else "NONE",
                    'calibrated_prediction': calibrated_pred,
                    'confidence': confidence,
                    'actual': actual_label,
                    'actual_return': actual_return,
                    'correct_raw': raw_pred == actual_label if raw_pred else False,
                    'correct_calibrated': calibrated_pred == actual_label,
                    'changed_by_calibration': calibrated_pred != raw_pred,
                    'response_preview': response[:200]
                })
        
        # Calculate and display metrics
        self.calculate_metrics(pair_results, single_results, calibration_stats, quick)
        
        # Save results
        if pair_results:
            pd.DataFrame(pair_results).to_csv('calibrated_pairs_results.csv', index=False)
            logger.info(f"Saved {len(pair_results)} pair results to calibrated_pairs_results.csv")
        
        if single_results:
            pd.DataFrame(single_results).to_csv('calibrated_singles_results.csv', index=False)
            logger.info(f"Saved {len(single_results)} single results to calibrated_singles_results.csv")
    
    def calculate_metrics(self, pair_results, single_results, calibration_stats, quick=False):
        """Calculate metrics with calibration analysis."""
        
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS (Threshold = {self.confidence_threshold})")
        print("="*60)
        
        # Pair metrics (unchanged)
        if pair_results:
            df_pairs = pd.DataFrame(pair_results)
            valid_pairs = df_pairs[df_pairs['actual'] != 'Unknown']
            
            if len(valid_pairs) > 0:
                print("\n📊 PAIR COMPARISON METRICS")
                print("-" * 40)
                
                accuracy = valid_pairs['correct'].mean()
                print(f"Overall Accuracy: {accuracy:.1%} ({len(valid_pairs)} pairs)")
                
                # By confidence
                for conf_level in [0.8, 0.7, 0.6]:
                    conf_pairs = valid_pairs[valid_pairs['confidence'] >= conf_level]
                    if len(conf_pairs) > 3:
                        conf_acc = conf_pairs['correct'].mean()
                        print(f"  Confidence ≥{conf_level}: {conf_acc:.1%} ({len(conf_pairs)} pairs)")
        
        # Single filing metrics with calibration analysis
        if single_results:
            df_singles = pd.DataFrame(single_results)
            valid_singles = df_singles[df_singles['actual'] != 'Unknown']
            
            if len(valid_singles) > 0:
                print("\n📊 SINGLE FILING METRICS - CALIBRATED")
                print("-" * 40)
                
                # Show calibration impact
                print(f"\n🎯 Calibration Impact:")
                print(f"  Threshold: {self.confidence_threshold}")
                print(f"  Predictions changed: {calibration_stats['changed']}/{calibration_stats['total']} ({100*calibration_stats['changed']/calibration_stats['total']:.1f}%)")
                
                # Compare raw vs calibrated accuracy
                raw_accuracy = valid_singles['correct_raw'].mean()
                calibrated_accuracy = valid_singles['correct_calibrated'].mean()
                
                print(f"\n📈 Accuracy Comparison:")
                print(f"  Raw Model: {raw_accuracy:.1%}")
                print(f"  Calibrated: {calibrated_accuracy:.1%}")
                print(f"  Improvement: {(calibrated_accuracy - raw_accuracy)*100:+.1f}%")
                
                # Calibrated confusion matrix
                print("\n🔍 Calibrated Confusion Matrix:")
                for actual in ['POSITIVE', 'NEGATIVE']:
                    for pred in ['POSITIVE', 'NEGATIVE']:
                        count = len(valid_singles[
                            (valid_singles['actual'] == actual) & 
                            (valid_singles['calibrated_prediction'] == pred)
                        ])
                        print(f"  Actual {actual}, Predicted {pred}: {count}")
                
                # Calculate calibrated metrics
                tp = len(valid_singles[(valid_singles['actual'] == 'POSITIVE') & 
                                       (valid_singles['calibrated_prediction'] == 'POSITIVE')])
                fp = len(valid_singles[(valid_singles['actual'] == 'NEGATIVE') & 
                                       (valid_singles['calibrated_prediction'] == 'POSITIVE')])
                fn = len(valid_singles[(valid_singles['actual'] == 'POSITIVE') & 
                                       (valid_singles['calibrated_prediction'] == 'NEGATIVE')])
                tn = len(valid_singles[(valid_singles['actual'] == 'NEGATIVE') & 
                                       (valid_singles['calibrated_prediction'] == 'NEGATIVE')])
                
                print("\n📊 Calibrated Classification Metrics:")
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                    print(f"  Precision (POSITIVE): {precision:.1%}")
                
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                    print(f"  Recall (POSITIVE): {recall:.1%}")
                
                if tn + fn > 0:
                    specificity = tn / (tn + fn)
                    print(f"  Specificity (NEGATIVE): {specificity:.1%}")
                
                if tp + fp > 0 and tp + fn > 0:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        print(f"  F1 Score: {f1:.3f}")
                
                # Show distribution of predictions
                pos_pred_count = len(valid_singles[valid_singles['calibrated_prediction'] == 'POSITIVE'])
                neg_pred_count = len(valid_singles[valid_singles['calibrated_prediction'] == 'NEGATIVE'])
                
                print(f"\n📊 Prediction Distribution (Calibrated):")
                print(f"  POSITIVE predictions: {pos_pred_count} ({100*pos_pred_count/len(valid_singles):.1f}%)")
                print(f"  NEGATIVE predictions: {neg_pred_count} ({100*neg_pred_count/len(valid_singles):.1f}%)")
                
                # Trading simulation with calibrated predictions
                print("\n💹 TRADING SIMULATION (Calibrated)")
                print("-" * 40)
                
                trades = valid_singles[valid_singles['calibrated_prediction'].isin(['POSITIVE', 'NEGATIVE'])].copy()
                if len(trades) > 0:
                    trades['position'] = trades['calibrated_prediction'].map({'POSITIVE': 1, 'NEGATIVE': -1})
                    trades['pnl'] = trades['position'] * trades['actual_return']
                    
                    total_return = trades['pnl'].sum()
                    avg_return = trades['pnl'].mean()
                    win_rate = (trades['pnl'] > 0).mean()
                    
                    if trades['pnl'].std() > 0:
                        sharpe = (avg_return * np.sqrt(252/5)) / trades['pnl'].std()
                    else:
                        sharpe = 0
                    
                    print(f"Number of Trades: {len(trades)}")
                    print(f"Win Rate: {win_rate:.1%}")
                    print(f"Average Return per Trade: {avg_return:.2%}")
                    print(f"Total Return: {total_return:.1%}")
                    print(f"Sharpe Ratio (annualized): {sharpe:.2f}")
                    
                    # Show confidence distribution for changed predictions
                    changed = valid_singles[valid_singles['changed_by_calibration'] == True]
                    if len(changed) > 0:
                        avg_conf_changed = changed['confidence'].mean()
                        print(f"\n🔄 Changed Predictions:")
                        print(f"  Average confidence of changed: {avg_conf_changed:.2f}")
        
        print("\n" + "="*60)
        if quick:
            print("Quick evaluation complete. Run without --quick for full results.")
        else:
            print("Full evaluation complete.")
        
        # Suggest optimal threshold
        if single_results and len(valid_singles) > 10:
            print("\n💡 Threshold Optimization Hint:")
            print(f"Current threshold: {self.confidence_threshold}")
            if calibrated_accuracy > raw_accuracy:
                print("✅ Calibration improved accuracy!")
            else:
                print("Consider trying different thresholds (0.6, 0.75, 0.8)")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Calibrated Binary Classification Evaluation')
    parser.add_argument('--model_path', default='./output_contrastive',
                       help='Path to fine-tuned model')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Confidence threshold for POSITIVE prediction (default: 0.7)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation with 50 samples')
    parser.add_argument('--verbose', action='store_true',
                       help='Show calibration changes')
    parser.add_argument('--test_path', default='data/test.json',
                       help='Path to test data')
    parser.add_argument('--metadata_path', default='data/test_metadata.json',
                       help='Path to test metadata')
    
    args = parser.parse_args()
    
    evaluator = CalibratedBinaryEvaluator(
        args.model_path, 
        confidence_threshold=args.threshold
    )
    
    evaluator.evaluate_test_set(
        test_path=args.test_path,
        metadata_path=args.metadata_path,
        quick=args.quick,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()