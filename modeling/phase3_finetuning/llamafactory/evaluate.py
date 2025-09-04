#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Qwen model
"""

import os
# Suppress transformers warnings about generation flags
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import List, Dict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class QwenEvaluator:
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """Initialize evaluator with fine-tuned model"""
        print(f"Loading base model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        print(f"Loading LoRA weights from: {model_path}")
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        # Define label mappings
        self.labels = [
            "Strong negative price movement expected (>3% decrease)",
            "Moderate negative price movement expected (0-3% decrease)",
            "Moderate positive price movement expected (0-3% increase)",
            "Strong positive price movement expected (>3% increase)"
        ]
        
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
    
    def predict(self, instruction: str, input_text: str, system: str = None) -> str:
        """Generate prediction for single input"""
        # Format prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": f"{instruction}\n\n{input_text}"})
        
        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  # Using greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def evaluate_dataset(self, data_path: str) -> Dict:
        """Evaluate on validation dataset"""
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        print(f"Evaluating on {len(data)} samples...")
        
        predictions = []
        true_labels = []
        
        for i, item in enumerate(data):
            if i % 10 == 0:
                print(f"Processing {i}/{len(data)}...")
            
            # Get prediction
            pred = self.predict(
                item["instruction"],
                item["input"],
                item.get("system", None)
            )
            
            # Map to label index
            pred_idx = self._map_to_label(pred)
            true_idx = self._map_to_label(item["output"])
            
            predictions.append(pred_idx)
            true_labels.append(true_idx)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(
            true_labels, 
            predictions,
            target_names=["STRONG_NEG", "NEG", "POS", "STRONG_POS"],
            output_dict=True
        )
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "predictions": predictions,
            "true_labels": true_labels
        }
    
    def _map_to_label(self, text: str) -> int:
        """Map text to label index"""
        text_lower = text.lower()
        
        if "strong" in text_lower and "negative" in text_lower:
            return 0
        elif "negative" in text_lower and "strong" not in text_lower:
            return 1
        elif "positive" in text_lower and "strong" not in text_lower:
            return 2
        elif "strong" in text_lower and "positive" in text_lower:
            return 3
        else:
            # Default to neutral/positive
            return 2

def main():
    """Main evaluation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./outputs/qwen2.5-1.5b-lora")
    parser.add_argument("--data_path", type=str, default="data/val.json")
    parser.add_argument("--output_path", type=str, default="evaluation_results.json")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QwenEvaluator(args.model_path)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.data_path)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    for label in ["STRONG_NEG", "NEG", "POS", "STRONG_POS"]:
        metrics = results['classification_report'][label]
        print(f"  {label:12} - P: {metrics['precision']:.3f}, R: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    
    print("\nConfusion Matrix:")
    print("  Predicted ->")
    print("  True ↓     S_NEG   NEG    POS   S_POS")
    for i, row in enumerate(results['confusion_matrix']):
        label = ["STRONG_NEG", "NEG", "POS", "STRONG_POS"][i]
        print(f"  {label:10}", end="")
        for val in row:
            print(f" {val:5}", end="")
        print()
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_path}")

if __name__ == "__main__":
    main()