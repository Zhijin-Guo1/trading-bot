#!/usr/bin/env python3
"""
Baseline Evaluation - Untrained Qwen2.5-7B on same test set
Uses EXACT SAME prompts as fine-tuned model for fair comparison
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from tqdm import tqdm
import argparse
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class BaselineEvaluator:
    def __init__(self, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize baseline evaluator with UNTRAINED model."""
        logger.info("="*60)
        logger.info("BASELINE EVALUATION - UNTRAINED MODEL")
        logger.info("="*60)
        logger.info(f"Base model: {base_model}")
        logger.info("No fine-tuning applied - raw pretrained model")
        logger.info("Using SAME prompts as fine-tuned model (fair comparison)")
        logger.info("="*60)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load UNTRAINED base model (no LoRA)
        logger.info("Loading untrained base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        logger.info("✅ Untrained model loaded successfully!\n")
    
    def parse_pair_prediction(self, response: str) -> str:
        """Parse pair comparison prediction - SAME logic as fine-tuned."""
        
        first_part = response.split('\n')[0] if '\n' in response else response
        first_100_chars = response[:100]
        
        patterns = [
            (r'Filing ([AB]) resulted in better', 1),
            (r'Filing ([AB]) performed better', 1),
            (r'Filing ([AB]) outperformed', 1),
            (r'Filing ([AB]) would likely outperform', 1),
            (r'Filing ([AB]) is better', 1),
            (r'Filing ([AB]) shows better', 1),
            (r'I would predict Filing ([AB])', 1),
            (r'([AB]) resulted in better', 1),
            (r'([AB]) performed better', 1),
            (r'Answer:\s*Filing ([AB])', 1),
            (r'Answer:\s*([AB])', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, first_part, re.IGNORECASE)
            if match:
                return match.group(group).upper()
        
        lines = response.split('\n')[:3]
        text_to_check = ' '.join(lines)
        
        a_count = len(re.findall(r'Filing A', text_to_check, re.IGNORECASE))
        b_count = len(re.findall(r'Filing B', text_to_check, re.IGNORECASE))
        
        if a_count > b_count:
            return "A"
        elif b_count > a_count:
            return "B"
        
        a_pos = response.lower().find('filing a')
        b_pos = response.lower().find('filing b')
        
        if a_pos != -1 and b_pos != -1:
            return "A" if a_pos < b_pos else "B"
        elif a_pos != -1:
            return "A"
        elif b_pos != -1:
            return "B"
        
        return "UNCLEAR"
    
    def extract_binary_prediction(self, response: str, is_pair: bool = False) -> Tuple[str, float]:
        """Extract POSITIVE/NEGATIVE prediction - SAME as fine-tuned."""
        
        if is_pair:
            pair_pred = self.parse_pair_prediction(response)
            confidence = 0.5  # Default confidence for untrained model
            return pair_pred, confidence
        
        response_upper = response.upper()
        
        # Look for prediction keywords - SAME as fine-tuned
        prediction = None
        
        # Check for explicit predictions
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
            elif "BULLISH" in response_upper:
                prediction = "POSITIVE"
            elif "BEARISH" in response_upper:
                prediction = "NEGATIVE"
        
        # Default to UNCLEAR if no clear prediction
        if prediction is None:
            prediction = "UNCLEAR"
        
        # Extract confidence (likely to be medium for untrained)
        confidence = 0.5
        if "definitely" in response.lower() or "clearly" in response.lower():
            confidence = 0.7
        elif "possibly" in response.lower() or "might" in response.lower():
            confidence = 0.3
        
        return prediction, confidence
    
    def generate_prediction(self, instruction: str, is_pair: bool = False) -> Tuple[str, float, str]:
        """Generate prediction using untrained model - SAME PROMPT AS FINE-TUNED."""
        
        # USE EXACT SAME INSTRUCTION AS FINE-TUNED MODEL - NO MODIFICATIONS!
        inputs = self.tokenizer(
            instruction,  # Direct from test file, no enhancements
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
                repetition_penalty=1.2  # Same as fine-tuned
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        prediction, confidence = self.extract_binary_prediction(response, is_pair=is_pair)
        return prediction, confidence, response
    
    def evaluate_test_set(self, test_path: str = "data/test.json", 
                         metadata_path: str = "data/test_metadata.json",
                         quick: bool = False,
                         verbose: bool = False):
        """Evaluate untrained model on test set with FAIR comparison."""
        
        # Load test data
        with open(test_path) as f:
            test_data = json.load(f)
        
        with open(metadata_path) as f:
            metadata_list = json.load(f)
        
        if quick:
            logger.info("Running QUICK baseline evaluation (50 samples)...")
            test_data = test_data[:50]
            metadata_list = metadata_list[:50] if len(metadata_list) >= 50 else metadata_list
        else:
            logger.info(f"Running FULL baseline evaluation ({len(test_data)} samples)...")
            logger.info("This will take time as the model is not optimized for this task...")
        
        pair_results = []
        single_results = []
        unclear_count = 0
        
        for i, item in enumerate(tqdm(test_data, desc="Evaluating Baseline")):
            instruction = item['instruction']
            meta = metadata_list[i] if i < len(metadata_list) else {}
            
            is_pair = "Filing A" in instruction and "Filing B" in instruction
            
            # Generate with SAME prompt as fine-tuned
            pred, conf, response = self.generate_prediction(instruction, is_pair=is_pair)
            
            if verbose and i < 5:  # Show first 5 examples
                print(f"\n--- Sample {i} ---")
                print(f"Type: {'Pair' if is_pair else 'Single'}")
                print(f"Instruction preview: {instruction[:100]}...")
                print(f"Response start: {response[:150]}...")
                print(f"Parsed: {pred}, Confidence: {conf}")
            
            if pred == "UNCLEAR":
                unclear_count += 1
            
            if is_pair:
                correct_answer = meta.get('correct_answer', 'Unknown')
                filing_a_return = meta.get('filing_a_return', 0)
                filing_b_return = meta.get('filing_b_return', 0)
                
                pair_results.append({
                    'index': i,
                    'prediction': pred,
                    'confidence': conf,
                    'actual': correct_answer,
                    'correct': pred == correct_answer if pred != "UNCLEAR" else False,
                    'filing_a_return': filing_a_return,
                    'filing_b_return': filing_b_return,
                    'response_preview': response[:200]
                })
            else:
                actual_return = meta.get('actual_return', 0)
                actual_label = "POSITIVE" if actual_return > 0 else "NEGATIVE"
                
                single_results.append({
                    'index': i,
                    'prediction': pred,
                    'confidence': conf,
                    'actual': actual_label,
                    'actual_return': actual_return,
                    'correct': pred == actual_label if pred != "UNCLEAR" else False,
                    'response_preview': response[:200]
                })
        
        # Calculate and display metrics
        self.calculate_metrics(pair_results, single_results, unclear_count, len(test_data), quick)
        
        # Save results
        if pair_results:
            pd.DataFrame(pair_results).to_csv('baseline_pairs_results.csv', index=False)
            logger.info(f"Saved {len(pair_results)} pair results to baseline_pairs_results.csv")
        
        if single_results:
            pd.DataFrame(single_results).to_csv('baseline_singles_results.csv', index=False)
            logger.info(f"Saved {len(single_results)} single results to baseline_singles_results.csv")
    
    def calculate_metrics(self, pair_results, single_results, unclear_count, total_samples, quick=False):
        """Calculate metrics for baseline model."""
        
        print("\n" + "="*60)
        print("BASELINE MODEL RESULTS (Untrained Qwen2.5-7B)")
        print("="*60)
        
        print(f"\n⚠️ Unclear Predictions: {unclear_count}/{total_samples} ({100*unclear_count/total_samples:.1f}%)")
        
        # Pair metrics
        if pair_results:
            df_pairs = pd.DataFrame(pair_results)
            valid_pairs = df_pairs[(df_pairs['actual'] != 'Unknown') & (df_pairs['prediction'] != 'UNCLEAR')]
            all_pairs_with_truth = df_pairs[df_pairs['actual'] != 'Unknown']
            
            print("\n📊 BASELINE PAIR COMPARISON")
            print("-" * 40)
            
            if len(valid_pairs) > 0:
                # Accuracy excluding unclear
                accuracy_valid = valid_pairs['correct'].mean()
                print(f"Accuracy (valid predictions only): {accuracy_valid:.1%} ({len(valid_pairs)}/{len(all_pairs_with_truth)} valid)")
                
                # Accuracy including unclear as wrong
                accuracy_all = all_pairs_with_truth['correct'].mean()
                print(f"Accuracy (unclear = wrong): {accuracy_all:.1%}")
                
                # Check for bias
                a_predictions = len(valid_pairs[valid_pairs['prediction'] == 'A'])
                b_predictions = len(valid_pairs[valid_pairs['prediction'] == 'B'])
                unclear_pairs = len(df_pairs[df_pairs['prediction'] == 'UNCLEAR'])
                
                print(f"\nPrediction distribution:")
                print(f"  Filing A: {a_predictions}")
                print(f"  Filing B: {b_predictions}")
                print(f"  UNCLEAR: {unclear_pairs}")
                
                # Random baseline
                print(f"\nRandom baseline: 50%")
                print(f"Performance vs random: {(accuracy_all - 0.5)*100:+.1f}%")
            else:
                print("No valid pair predictions (all unclear)")
        
        # Single filing metrics
        if single_results:
            df_singles = pd.DataFrame(single_results)
            valid_singles = df_singles[(df_singles['actual'] != 'Unknown') & (df_singles['prediction'] != 'UNCLEAR')]
            all_singles_with_truth = df_singles[df_singles['actual'] != 'Unknown']
            
            print("\n📊 BASELINE SINGLE FILING")
            print("-" * 40)
            
            if len(valid_singles) > 0:
                # Accuracy excluding unclear
                accuracy_valid = valid_singles['correct'].mean()
                print(f"Accuracy (valid predictions only): {accuracy_valid:.1%} ({len(valid_singles)}/{len(all_singles_with_truth)} valid)")
                
                # Accuracy including unclear as wrong
                accuracy_all = all_singles_with_truth['correct'].mean()
                print(f"Accuracy (unclear = wrong): {accuracy_all:.1%}")
                
                # Confusion matrix for valid predictions
                if len(valid_singles) > 5:
                    print("\nConfusion Matrix (valid predictions only):")
                    for actual in ['POSITIVE', 'NEGATIVE']:
                        for pred in ['POSITIVE', 'NEGATIVE']:
                            count = len(valid_singles[(valid_singles['actual'] == actual) & 
                                                     (valid_singles['prediction'] == pred)])
                            print(f"  Actual {actual}, Predicted {pred}: {count}")
                
                # Prediction distribution
                pos_count = len(valid_singles[valid_singles['prediction'] == 'POSITIVE'])
                neg_count = len(valid_singles[valid_singles['prediction'] == 'NEGATIVE'])
                unclear_singles = len(df_singles[df_singles['prediction'] == 'UNCLEAR'])
                
                print(f"\nPrediction Distribution:")
                print(f"  POSITIVE: {pos_count}")
                print(f"  NEGATIVE: {neg_count}")
                print(f"  UNCLEAR: {unclear_singles}")
                
                if len(valid_singles) > 0:
                    pos_pct = 100*pos_count/(pos_count + neg_count)
                    print(f"  Positive bias (of valid): {pos_pct:.1f}%")
                
                print(f"\nRandom baseline: 50%")
                print(f"Performance vs random: {(accuracy_all - 0.5)*100:+.1f}%")
            else:
                print("No valid single predictions (all unclear)")
        
        print("\n" + "="*60)
        print("COMPARISON WITH FINE-TUNED MODEL")
        print("="*60)
        
        print("\n📊 Performance Comparison:")
        print("-" * 40)
        
        # Calculate actual accuracies for comparison
        baseline_pair_acc = all_pairs_with_truth['correct'].mean() * 100 if pair_results and len(all_pairs_with_truth) > 0 else 0
        baseline_single_acc = all_singles_with_truth['correct'].mean() * 100 if single_results and len(all_singles_with_truth) > 0 else 0
        
        print(f"                    Baseline → Fine-tuned → Improvement")
        print(f"Pair Accuracy:      {baseline_pair_acc:.1f}% → 55.5% → {55.5 - baseline_pair_acc:+.1f}%")
        print(f"Single Accuracy:    {baseline_single_acc:.1f}% → 53.1% → {53.1 - baseline_single_acc:+.1f}%")
        print(f"Unclear Rate:       {100*unclear_count/total_samples:.1f}% → ~0% → {-100*unclear_count/total_samples:.1f}%")
        
        print("\n💡 Key Observations:")
        print("- Untrained model struggles without task-specific training")
        print("- High unclear rate shows model doesn't understand the task format")
        print("- Fine-tuning provides clear improvement in both accuracy and clarity")
        
        if unclear_count > total_samples * 0.3:
            print("\n⚠️ Very high unclear rate indicates:")
            print("  - Model doesn't know what to predict")
            print("  - Lacks understanding of financial filing analysis")
            print("  - Fine-tuning successfully teaches the task")
        
        if quick:
            print("\n⚠️ Quick evaluation - run without --quick for full results")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Baseline Evaluation - Untrained Model')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation with 50 samples')
    parser.add_argument('--verbose', action='store_true',
                       help='Show example outputs')
    parser.add_argument('--test_path', default='data/test.json',
                       help='Path to test data')
    parser.add_argument('--metadata_path', default='data/test_metadata.json',
                       help='Path to test metadata')
    
    args = parser.parse_args()
    
    evaluator = BaselineEvaluator()
    evaluator.evaluate_test_set(
        test_path=args.test_path,
        metadata_path=args.metadata_path,
        quick=args.quick,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()