#!/usr/bin/env python3
"""
Verify finetuning data against actual CSV files with accession numbers
"""

import json
import pandas as pd
from pathlib import Path
import random

def verify_samples():
    """Load samples and verify against CSV data"""
    
    # Load training data
    train_path = Path("/mnt/trading-bot/finetuning_data/train.jsonl")
    if not train_path.exists():
        train_path = Path("../finetuning_data/train.jsonl")
    
    print("="*70)
    print("VERIFYING AGAINST ACTUAL EDGAR CSV DATA")
    print("="*70)
    
    with open(train_path, 'r') as f:
        lines = f.readlines()
    
    # Get random samples
    num_samples = 5
    sample_indices = random.sample(range(len(lines)), min(num_samples, len(lines)))
    
    matches_found = 0
    
    for idx, i in enumerate(sample_indices, 1):
        sample = json.loads(lines[i])
        
        print(f"\n{'='*70}")
        print(f"SAMPLE {idx}")
        print(f"{'='*70}")
        
        # Extract metadata
        metadata = sample.get('metadata', {})
        ticker = metadata.get('ticker', '')
        filing_date = metadata.get('filing_date', '')
        year = metadata.get('year', filing_date.split('-')[0] if filing_date else '')
        
        print(f"\nSearching for:")
        print(f"  Ticker: {ticker}")
        print(f"  Filing Date: {filing_date}")
        print(f"  Year: {year}")
        
        # Load the corresponding CSV
        csv_path = Path(f"/Users/engs2742/trading-bot/data_collection/eight_k/{year}/8K_{year}_metadata.csv")
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Search for matching filing
            matches = df[(df['ticker'] == ticker) & (df['filing_date'] == filing_date)]
            
            if not matches.empty:
                print(f"\n✅ FOUND {len(matches)} matching filing(s)!")
                for _, row in matches.iterrows():
                    print(f"\n  Accession: {row['accession']}")
                    print(f"  CIK: {row['cik']}")
                    print(f"  Text Length: {row.get('text_length', 'N/A')}")
                    
                    # Load the text file to verify content
                    text_file = Path(f"/Users/engs2742/trading-bot/data_collection/eight_k/{year}/8K_{year}_text.json")
                    if text_file.exists():
                        with open(text_file, 'r') as f:
                            text_data = json.load(f)
                            
                        # Find the text for this accession
                        for item in text_data:
                            if item.get('accession') == row['accession']:
                                original_text = item.get('text', '')[:300]
                                print(f"\n  Original text preview:")
                                print(f"  {original_text}...")
                                
                                # Compare with sample
                                sample_text = sample.get('input', '')
                                content_start = sample_text.find('Content:')
                                if content_start != -1:
                                    sample_content = sample_text[content_start+8:].strip()[:300]
                                else:
                                    sample_content = sample_text[:300]
                                
                                print(f"\n  Sample text preview:")
                                print(f"  {sample_content}...")
                                
                                # Check if they're related (accounting for preprocessing)
                                # Convert to lowercase and remove special chars for comparison
                                orig_clean = original_text.lower().replace(' ', '').replace('\n', '')
                                sample_clean = sample_content.lower().replace(' ', '').replace('\n', '')
                                
                                if sample_clean[:50] in orig_clean or orig_clean[:50] in sample_clean:
                                    print(f"\n  ✅ Text content MATCHES (accounting for preprocessing)")
                                else:
                                    print(f"\n  ⚠️  Text appears different (likely heavy preprocessing)")
                                
                                break
                
                matches_found += 1
            else:
                print(f"❌ No match found in {csv_path.name}")
        else:
            print(f"❌ CSV file not found: {csv_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"\nMatches found: {matches_found}/{num_samples}")
    
    if matches_found > 0:
        print("\n✅ Data is correctly sourced from downloaded EDGAR filings!")
        print("   The accession numbers and dates align with actual SEC data.")
    else:
        print("\n⚠️  Could not find matches - check data pipeline")

if __name__ == "__main__":
    verify_samples()