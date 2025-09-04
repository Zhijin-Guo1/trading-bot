#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen model
Use this to make predictions on new 8-K filings
"""

import os
import json
import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_model(model_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load fine-tuned model for inference"""
    print(f"Loading model from {model_path}")
    
    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    return model, tokenizer

def create_prompt(filing_data: dict, label_type: str = 'binary') -> str:
    """Create prompt from filing data"""
    
    momentum_info = f"""
Recent Performance:
- 7-day momentum: {filing_data.get('momentum_7d', 0):.2f}%
- 30-day momentum: {filing_data.get('momentum_30d', 0):.2f}%
- 90-day momentum: {filing_data.get('momentum_90d', 0):.2f}%"""
    
    market_info = f"VIX (volatility): {filing_data.get('vix_level', 0):.2f}" if 'vix_level' in filing_data else ""
    
    if label_type == 'binary':
        instruction = """You are a financial analyst predicting stock movements based on 8-K filings.
Analyze the following 8-K filing and market context to predict if the stock will go UP or DOWN in the next trading day.

IMPORTANT: Respond with ONLY one word: either "UP" or "DOWN"."""
    else:
        instruction = """You are a financial analyst predicting stock movements based on 8-K filings.
Analyze the following 8-K filing and market context to predict if the stock will go UP (>1%), DOWN (<-1%), or STAY (-1% to 1%) in the next trading day.

IMPORTANT: Respond with ONLY one word: either "UP", "DOWN", or "STAY"."""
    
    context = f"""
Stock: {filing_data.get('ticker', 'N/A')}
Sector: {filing_data.get('sector', 'N/A')}
Industry: {filing_data.get('industry', 'N/A')}
Filing Date: {filing_data.get('filing_date', 'N/A')}
8-K Items: {filing_data.get('items_present', 'N/A')}

{momentum_info}
{market_info}

8-K Filing Summary:
{filing_data.get('summary', 'No summary available')}
"""
    
    full_prompt = f"{instruction}\n{context}\n\nPrediction:"
    
    return full_prompt

def predict(model, tokenizer, filing_data: dict, label_type: str = 'binary') -> str:
    """Make prediction for a single filing"""
    
    # Create prompt
    prompt = create_prompt(filing_data, label_type)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode prediction
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = full_response.replace(prompt, '').strip().split()[0].upper()
    
    # Validate prediction
    valid_labels = ['UP', 'DOWN'] if label_type == 'binary' else ['UP', 'DOWN', 'STAY']
    if prediction not in valid_labels:
        prediction = valid_labels[0]  # Default to first valid label
    
    return prediction

def batch_predict(model, tokenizer, data_path: str, label_type: str = 'binary', max_samples: int = None):
    """Make predictions for a batch of filings"""
    
    # Load data
    if data_path.endswith('.json'):
        data = pd.read_json(data_path)
    else:
        data = pd.read_csv(data_path)
    
    if max_samples:
        data = data.head(max_samples)
    
    print(f"Making predictions for {len(data)} samples...")
    
    predictions = []
    for idx, row in data.iterrows():
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(data)} samples...")
        
        filing_data = row.to_dict()
        pred = predict(model, tokenizer, filing_data, label_type)
        predictions.append(pred)
    
    # Add predictions to dataframe
    data['prediction'] = predictions
    
    # Calculate accuracy if true labels exist
    if 'binary_label' in data.columns and label_type == 'binary':
        accuracy = (data['prediction'] == data['binary_label'].str.upper()).mean()
        print(f"\nAccuracy: {accuracy:.4f}")
    elif 'three_class_label' in data.columns and label_type == 'three_class':
        accuracy = (data['prediction'] == data['three_class_label'].str.upper()).mean()
        print(f"\nAccuracy: {accuracy:.4f}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Inference with fine-tuned Qwen model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Base model name')
    parser.add_argument('--data_path', type=str,
                        help='Path to data file for batch prediction')
    parser.add_argument('--label_type', type=str, default='binary',
                        choices=['binary', 'three_class'],
                        help='Classification type')
    parser.add_argument('--output_path', type=str,
                        help='Path to save predictions')
    parser.add_argument('--max_samples', type=int,
                        help='Maximum samples to process')
    
    # For single prediction
    parser.add_argument('--ticker', type=str,
                        help='Stock ticker for single prediction')
    parser.add_argument('--summary', type=str,
                        help='8-K summary for single prediction')
    parser.add_argument('--filing_date', type=str,
                        help='Filing date for single prediction')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    if args.data_path:
        # Batch prediction
        results = batch_predict(
            model, 
            tokenizer, 
            args.data_path, 
            args.label_type,
            args.max_samples
        )
        
        if args.output_path:
            if args.output_path.endswith('.json'):
                results.to_json(args.output_path, orient='records', indent=2)
            else:
                results.to_csv(args.output_path, index=False)
            print(f"Predictions saved to {args.output_path}")
    
    elif args.summary:
        # Single prediction
        filing_data = {
            'ticker': args.ticker or 'UNKNOWN',
            'summary': args.summary,
            'filing_date': args.filing_date or 'N/A',
            'momentum_7d': 0,
            'momentum_30d': 0,
            'momentum_90d': 0,
            'vix_level': 15  # Default VIX
        }
        
        prediction = predict(model, tokenizer, filing_data, args.label_type)
        print(f"\nPrediction: {prediction}")
    
    else:
        print("Please provide either --data_path for batch prediction or --summary for single prediction")

if __name__ == '__main__':
    main()