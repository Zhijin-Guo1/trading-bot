#!/usr/bin/env python3
"""
Check label consistency between training and validation data
"""

import json
from collections import Counter

def check_labels():
    # Check training data
    print("="*70)
    print("TRAINING DATA LABELS")
    print("="*70)
    
    with open('data/train.json', 'r') as f:
        train_data = json.load(f)
    
    train_outputs = [item['output'] for item in train_data]
    train_counter = Counter(train_outputs)
    
    print(f"\nTotal training samples: {len(train_outputs)}")
    print(f"Unique labels: {len(train_counter)}")
    print("\nLabel distribution:")
    for label, count in train_counter.most_common():
        print(f"  {count:4d} - {label}")
    
    # Check validation data
    print("\n" + "="*70)
    print("VALIDATION DATA LABELS")
    print("="*70)
    
    with open('data/val.json', 'r') as f:
        val_data = json.load(f)
    
    val_outputs = [item['output'] for item in val_data]
    val_counter = Counter(val_outputs)
    
    print(f"\nTotal validation samples: {len(val_outputs)}")
    print(f"Unique labels: {len(val_counter)}")
    print("\nLabel distribution:")
    for label, count in val_counter.most_common():
        print(f"  {count:4d} - {label}")
    
    # Check for mismatches
    print("\n" + "="*70)
    print("LABEL CONSISTENCY CHECK")
    print("="*70)
    
    train_labels = set(train_counter.keys())
    val_labels = set(val_counter.keys())
    
    if train_labels == val_labels:
        print("\n✅ Training and validation labels are consistent!")
    else:
        print("\n❌ MISMATCH DETECTED!")
        
        if train_labels - val_labels:
            print("\nLabels in training but NOT in validation:")
            for label in train_labels - val_labels:
                print(f"  - {label}")
        
        if val_labels - train_labels:
            print("\nLabels in validation but NOT in training:")
            for label in val_labels - train_labels:
                print(f"  - {label}")
    
    # Check original data format
    print("\n" + "="*70)
    print("CHECKING ORIGINAL DATA FORMAT")
    print("="*70)
    
    orig_train_path = "/mnt/trading-bot/finetuning_data/train.jsonl"
    try:
        with open(orig_train_path, 'r') as f:
            line = f.readline()
            sample = json.loads(line)
            print(f"\nOriginal format sample:")
            print(f"  Keys: {list(sample.keys())}")
            if 'output' in sample:
                print(f"  Output: {sample['output']}")
            if 'label' in sample:
                print(f"  Label: {sample['label']}")
    except FileNotFoundError:
        print(f"Original file not found at {orig_train_path}")

if __name__ == "__main__":
    check_labels()