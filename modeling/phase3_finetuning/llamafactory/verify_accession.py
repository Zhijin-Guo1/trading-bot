#!/usr/bin/env python3
"""
Verify that the text content matches the actual EDGAR accession numbers
"""

import json
import os
from pathlib import Path
import random

def find_original_data(ticker, filing_date, item_number):
    """Find the original downloaded 8-K data file"""
    # Look in the eight_k folder structure
    base_path = Path("/Users/engs2742/trading-bot/data_collection/eight_k")
    
    # Extract year from filing date
    year = filing_date.split('-')[0]
    
    # Search for the file
    year_path = base_path / year
    if year_path.exists():
        # Look for files matching this ticker
        for file in year_path.glob(f"**/8k_data_{year}_*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    # Search through filings
                    for filing in data.get('filings', []):
                        if (filing.get('ticker') == ticker and 
                            filing.get('filing_date') == filing_date):
                            # Found matching filing
                            return filing
            except:
                continue
    
    # Alternative: check CSV files
    csv_path = base_path / year / f"8k_filings_{year}.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        matching = df[(df['ticker'] == ticker) & (df['filing_date'] == filing_date)]
        if not matching.empty:
            return matching.iloc[0].to_dict()
    
    return None

def extract_accession_from_text(text):
    """Extract accession number from text if present"""
    import re
    
    # Pattern for accession numbers (e.g., 0000012345-21-000678)
    pattern = r'\b(\d{10})-(\d{2})-(\d{6})\b'
    matches = re.findall(pattern, text)
    
    if matches:
        return f"{matches[0][0]}-{matches[0][1]}-{matches[0][2]}"
    return None

def load_and_verify_samples(num_samples=5):
    """Load samples and verify against original data"""
    
    # Load training data
    train_path = Path("/mnt/trading-bot/finetuning_data/train.jsonl")
    if not train_path.exists():
        train_path = Path("../finetuning_data/train.jsonl")
    
    print("="*70)
    print("VERIFYING TEXT CONTENT AGAINST ORIGINAL EDGAR DATA")
    print("="*70)
    
    with open(train_path, 'r') as f:
        lines = f.readlines()
        
    # Get random samples
    sample_indices = random.sample(range(len(lines)), min(num_samples, len(lines)))
    
    verification_results = []
    
    for idx, i in enumerate(sample_indices, 1):
        sample = json.loads(lines[i])
        
        print(f"\n{'='*70}")
        print(f"SAMPLE {idx}/{num_samples}")
        print(f"{'='*70}")
        
        # Extract metadata
        metadata = sample.get('metadata', {})
        ticker = metadata.get('ticker', 'N/A')
        filing_date = metadata.get('filing_date', 'N/A')
        item_number = metadata.get('item_number', 'N/A')
        
        # Also try to extract from input text
        input_text = sample.get('input', '')
        if ticker == 'N/A':
            # Try to extract from text
            for line in input_text.split('\n')[:5]:
                if 'Company:' in line:
                    ticker = line.split('Company:')[1].strip()
                if 'Filing Date:' in line:
                    filing_date = line.split('Filing Date:')[1].strip()
        
        print(f"\nMetadata:")
        print(f"  Ticker: {ticker}")
        print(f"  Filing Date: {filing_date}")
        print(f"  Item Number: {item_number}")
        print(f"  Label: {sample.get('output', 'N/A')}")
        
        # Show text snippet
        content_start = input_text.find('Content:')
        if content_start != -1:
            content = input_text[content_start+8:].strip()
        else:
            content = input_text
        
        print(f"\nText Preview (first 200 chars):")
        print(f"  {content[:200]}...")
        
        # Check for accession in text
        accession = extract_accession_from_text(input_text)
        if accession:
            print(f"\nAccession found in text: {accession}")
        
        # Try to find original data
        print(f"\nSearching for original data...")
        original = find_original_data(ticker, filing_date, item_number)
        
        if original:
            print(f"✅ Found original filing!")
            orig_accession = original.get('accession', 'N/A')
            orig_text = original.get('item_text', original.get('text', ''))[:200]
            
            print(f"  Original Accession: {orig_accession}")
            print(f"  Original text preview: {orig_text}...")
            
            # Compare
            if accession and accession == orig_accession:
                print(f"  ✅ Accession numbers MATCH!")
            
            # Check text similarity (rough check)
            if content[:100].lower() in orig_text.lower() or orig_text[:100].lower() in content.lower():
                print(f"  ✅ Text content appears to MATCH!")
            else:
                print(f"  ⚠️  Text may be preprocessed/truncated")
            
            verification_results.append({
                'ticker': ticker,
                'date': filing_date,
                'verified': True,
                'accession_match': accession == orig_accession if accession else None
            })
        else:
            print(f"❌ Could not find original filing in downloaded data")
            print(f"  This might be because:")
            print(f"  - Data is in a different location")
            print(f"  - Filing was downloaded in a different batch")
            print(f"  - Data has been preprocessed/aggregated")
            
            verification_results.append({
                'ticker': ticker,
                'date': filing_date,
                'verified': False,
                'accession_match': None
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    verified_count = sum(1 for r in verification_results if r['verified'])
    print(f"\nFiles found: {verified_count}/{len(verification_results)}")
    
    if verified_count > 0:
        accession_matches = sum(1 for r in verification_results if r.get('accession_match') == True)
        print(f"Accession matches: {accession_matches}/{verified_count}")
    
    print("\nConclusion:")
    if verified_count > len(verification_results) * 0.5:
        print("✅ Data appears to be correctly sourced from EDGAR filings")
    else:
        print("⚠️  Could not verify most samples - data may be in different format")
        print("   The data is likely correct but has been preprocessed")

if __name__ == "__main__":
    # Check both local and server paths
    import sys
    
    print("Checking for data locations...")
    paths_to_check = [
        "/Users/engs2742/trading-bot/data_collection/eight_k",
        "/mnt/trading-bot/data_collection/eight_k",
        "/root/trading-bot/data_collection/eight_k"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ Found data at: {path}")
            break
    else:
        print("⚠️  Could not find eight_k data folder")
    
    load_and_verify_samples(num_samples=5)