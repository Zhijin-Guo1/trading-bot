#!/usr/bin/env python3
"""
Filter out routine_filing entries from enhanced datasets
"""

import pandas as pd
import os
from pathlib import Path

def filter_datasets():
    """Filter out routine_filing rows from train, val, and test datasets"""
    
    # Create filtered_data directory if it doesn't exist
    filtered_dir = Path("filtered_data")
    filtered_dir.mkdir(exist_ok=True)
    
    # Process each file
    files = ['enhanced_train.csv', 'enhanced_val.csv', 'enhanced_test.csv']
    
    for file in files:
        print(f"\nProcessing {file}...")
        
        # Read the dataset
        df = pd.read_csv(file)
        print(f"  Original shape: {df.shape}")
        
        # Check if sub_topic column exists
        if 'sub_topic' not in df.columns:
            print(f"  WARNING: 'sub_topic' column not found in {file}")
            continue
        
        # Count routine_filing entries before filtering
        routine_count = (df['sub_topic'] == 'routine_filing').sum()
        print(f"  Routine filings found: {routine_count}")
        
        # Filter out routine_filing entries
        filtered_df = df[df['sub_topic'] != 'routine_filing'].copy()
        print(f"  Filtered shape: {filtered_df.shape}")
        print(f"  Rows removed: {len(df) - len(filtered_df)}")
        
        # Save filtered data
        output_file = filtered_dir / file.replace('enhanced_', 'filtered_')
        filtered_df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        
        # Print sub_topic distribution after filtering
        print(f"  Sub-topic distribution after filtering:")
        sub_topic_counts = filtered_df['sub_topic'].value_counts().head(10)
        for topic, count in sub_topic_counts.items():
            print(f"    {topic}: {count} ({count/len(filtered_df)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("FILTERING COMPLETE")
    print("="*60)
    print(f"Filtered datasets saved in: {filtered_dir.absolute()}")

if __name__ == "__main__":
    filter_datasets()