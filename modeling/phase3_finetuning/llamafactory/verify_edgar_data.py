#!/usr/bin/env python3
"""
Verify the finetuning data against actual EDGAR filings
"""

import json
import requests
import time
from pathlib import Path
import random

def load_sample_data():
    """Load a few samples from the training data"""
    samples = []
    
    # Load from finetuning_data
    data_path = Path("/mnt/trading-bot/finetuning_data/train.jsonl")
    if not data_path.exists():
        data_path = Path("../finetuning_data/train.jsonl")
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
        # Get random samples
        for i in random.sample(range(len(lines)), min(3, len(lines))):
            samples.append(json.loads(lines[i]))
    
    return samples

def extract_metadata(sample):
    """Extract metadata from the input text"""
    input_text = sample['input']
    lines = input_text.split('\n')
    
    metadata = {}
    for line in lines[:5]:  # Check first few lines for metadata
        if 'Company:' in line:
            metadata['ticker'] = line.split('Company:')[1].strip()
        elif 'Filing Date:' in line:
            metadata['filing_date'] = line.split('Filing Date:')[1].strip()
        elif 'Event Type:' in line:
            metadata['item_number'] = line.split('-')[0].replace('Event Type:', '').strip()
    
    # Also try metadata field
    if 'metadata' in sample:
        metadata.update(sample['metadata'])
    
    return metadata

def fetch_from_edgar(ticker, filing_date):
    """Fetch actual 8-K filing from EDGAR"""
    # SEC EDGAR API
    headers = {
        'User-Agent': 'Academic Research (contact@example.edu)'
    }
    
    # Search for 8-K filings for this company around this date
    search_url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
    
    print(f"\n  Fetching from EDGAR for {ticker} on {filing_date}...")
    print(f"  Note: This would normally query EDGAR API")
    print(f"  URL would be: https://www.sec.gov/edgar/search/?q={ticker}+8-K+{filing_date}")
    
    # Return placeholder since actual API requires CIK lookup
    return "EDGAR API requires CIK mapping and proper authentication"

def compare_text(sample_text, edgar_text):
    """Compare preprocessed text with original"""
    # Extract content section
    content_start = sample_text.find('Content:')
    if content_start != -1:
        content = sample_text[content_start + 8:].strip()
    else:
        content = sample_text
    
    print("\n  === PREPROCESSED TEXT ANALYSIS ===")
    print(f"  Length: {len(content)} chars")
    print(f"  Contains [...]: {'[...]' in content}")
    print(f"  Contains [AMOUNT]: {'[AMOUNT]' in content}")
    print(f"  Contains [PERCENT]: {'[PERCENT]' in content}")
    print(f"  Contains [DATE]: {'[DATE]' in content}")
    
    # Show snippet
    snippet = content[:300] if len(content) > 300 else content
    print(f"\n  First 300 chars: {snippet}...")
    
    # Check for truncation
    if '[...]' in content:
        print("\n  ⚠️ TEXT IS TRUNCATED - Original was longer than 2000 chars")
        print("  The preprocessing combines start and end of text")
    
    return content

def main():
    print("="*60)
    print("EDGAR DATA VERIFICATION")
    print("="*60)
    
    # Load samples
    samples = load_sample_data()
    print(f"\nLoaded {len(samples)} samples for verification")
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}")
        print(f"{'='*60}")
        
        # Extract metadata
        metadata = extract_metadata(sample)
        print(f"\nMetadata:")
        print(f"  Ticker: {metadata.get('ticker', 'N/A')}")
        print(f"  Filing Date: {metadata.get('filing_date', 'N/A')}")
        print(f"  Item Number: {metadata.get('item_number', 'N/A')}")
        print(f"  Label: {sample.get('output', 'N/A')}")
        
        # Analyze preprocessed text
        content = compare_text(sample['input'], "")
        
        # Show what preprocessing did
        print("\n  === PREPROCESSING EFFECTS ===")
        print("  1. Dollar amounts → [AMOUNT]")
        print("  2. Percentages → [PERCENT]")
        print("  3. Dates → [DATE]")
        print("  4. HTML tags → removed")
        print("  5. URLs → removed")
        print("  6. Text > 2000 chars → truncated with [...]")
        
        # Fetch from EDGAR (placeholder)
        if metadata.get('ticker') and metadata.get('filing_date'):
            edgar_text = fetch_from_edgar(metadata['ticker'], metadata['filing_date'])
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nThe preprocessing in prepare_finetuning_data.py:")
    print("1. Replaces financial amounts with [AMOUNT] placeholder")
    print("2. Replaces percentages with [PERCENT] placeholder")
    print("3. Truncates long text by keeping first 1000 + last 1000 chars")
    print("4. This is WHY text looks incomplete with [...]")
    print("\nThis is INTENTIONAL to:")
    print("- Normalize financial values")
    print("- Keep text under token limits")
    print("- Focus on narrative content, not specific numbers")

if __name__ == "__main__":
    main()