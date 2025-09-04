#!/usr/bin/env python3
"""
Fix data format for LlamaFactory - ensure outputs are properly included
"""

import json
from pathlib import Path
from typing import Dict, List

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping line due to error: {e}")
    return data

def create_proper_format(item: Dict) -> Dict:
    """Convert to proper LlamaFactory format with actual outputs"""
    
    # Extract the actual label/output from original data
    output = item.get("output", item.get("label", ""))
    
    # Map to human-readable predictions
    output_mapping = {
        "STRONG_POSITIVE": "Strong positive price movement expected",
        "POSITIVE": "Moderate positive price movement expected",
        "NEGATIVE": "Moderate negative price movement expected",
        "STRONG_NEGATIVE": "Strong negative price movement expected"
    }
    
    # Get the mapped output or use original
    final_output = output_mapping.get(output, output)
    
    # Extract instruction and input
    instruction = item.get("instruction", "Analyze this 8-K filing and predict stock movement")
    input_text = item.get("input", "")
    
    # Truncate input if too long
    if len(input_text) > 2000:
        input_text = input_text[:2000] + "..."
    
    # Create the formatted item
    return {
        "instruction": instruction,
        "input": input_text,
        "output": final_output,  # This is the key - must have actual output!
        "system": "You are a financial analyst specializing in SEC filings analysis. Predict stock price movements based on 8-K filings."
    }

def verify_data(data: List[Dict], name: str):
    """Verify data has proper outputs"""
    empty_outputs = 0
    sample_outputs = []
    
    for i, item in enumerate(data[:100]):  # Check first 100
        if not item.get("output") or item["output"] == "":
            empty_outputs += 1
        elif i < 5:  # Show first 5 outputs
            sample_outputs.append(item["output"])
    
    print(f"\n{name} Data Verification:")
    print(f"  Total samples: {len(data)}")
    print(f"  Empty outputs in first 100: {empty_outputs}")
    print(f"  Sample outputs: {sample_outputs[:3]}")

def main():
    """Fix the data format"""
    
    # Find data source
    base_dir = Path(__file__).parent
    data_source = Path("/mnt/trading-bot/finetuning_data")
    
    if not data_source.exists():
        print(f"Error: Data source not found at {data_source}")
        return
    
    # Create output directory
    output_dir = base_dir / "data"
    output_dir.mkdir(exist_ok=True)
    
    # Process train data
    train_file = data_source / "train.jsonl"
    if train_file.exists():
        print(f"Processing {train_file}...")
        train_data = load_jsonl(train_file)
        train_formatted = [create_proper_format(item) for item in train_data]
        
        # Verify before saving
        verify_data(train_formatted, "Training")
        
        # Save
        with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_formatted, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(train_formatted)} training samples to {output_dir}/train.json")
    
    # Process validation data
    val_file = data_source / "val.jsonl"
    if val_file.exists():
        print(f"\nProcessing {val_file}...")
        val_data = load_jsonl(val_file)
        val_formatted = [create_proper_format(item) for item in val_data]
        
        # Verify before saving
        verify_data(val_formatted, "Validation")
        
        # Save
        with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
            json.dump(val_formatted, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(val_formatted)} validation samples to {output_dir}/val.json")
    
    # Create dataset_info.json
    dataset_info = {
        "8k_train": {
            "file_name": "train.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system"
            }
        },
        "8k_val": {
            "file_name": "val.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system"
            }
        }
    }
    
    with open(output_dir / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Show a complete example
    if train_formatted:
        print("\n" + "="*50)
        print("SAMPLE TRAINING EXAMPLE:")
        print("="*50)
        sample = train_formatted[0]
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Input: {sample['input'][:200]}...")
        print(f"Output: {sample['output']}")  # Full output - this is what model learns!
        print(f"System: {sample['system'][:100]}...")

if __name__ == "__main__":
    main()