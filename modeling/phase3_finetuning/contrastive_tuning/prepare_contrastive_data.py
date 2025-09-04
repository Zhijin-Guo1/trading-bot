#!/usr/bin/env python3
"""
Prepare Contrastive Learning Data for LlamaFactory Fine-tuning
==============================================================
This script creates pairs of 8-K filings for contrastive learning.
Each pair consists of a high-performing and low-performing filing 
from the same event type.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Event types to process - using only metadata, no external metrics
# Using float keys to match the data type in the CSV files
EVENT_TYPES = {
    7.01: {"name": "Regulation FD Disclosure", "threshold": 2.5},
    9.01: {"name": "Financial Statements", "threshold": 2.5},
    5.07: {"name": "Shareholder Matters", "threshold": 2.5},
    5.02: {"name": "Officer Changes", "threshold": 2.5},
    1.01: {"name": "Material Agreements", "threshold": 2.5},
    2.02: {"name": "Results of Operations", "threshold": 2.0}
}

class ContrastiveDataPreparator:
    def __init__(self, data_path: str = "../../../data_collection/filtered_profitable_data"):
        """Initialize the data preparator."""
        self.data_path = Path(data_path)
        self.output_path = Path("data")
        self.output_path.mkdir(exist_ok=True)
        
        # Load all data
        self.metadata_df = None
        self.items_df = None
        self.load_data()
        
    def load_data(self):
        """Load filtered profitable data from all years."""
        logger.info("Loading data from filtered_profitable_data...")
        
        # Load combined data if available
        metadata_path = self.data_path / "all_profitable_metadata.csv"
        items_path = self.data_path / "all_profitable_items.csv"
        
        if metadata_path.exists() and items_path.exists():
            self.metadata_df = pd.read_csv(metadata_path)
            self.items_df = pd.read_csv(items_path)
        else:
            # Load year by year
            all_metadata = []
            all_items = []
            
            for year in ["2021", "2022", "2023", "2024"]:
                year_dir = self.data_path / year
                if year_dir.exists():
                    meta_path = year_dir / "items_metadata_enhanced.csv"
                    items_path = year_dir / "items_separated_profitable.csv"
                    
                    if meta_path.exists():
                        metadata = pd.read_csv(meta_path)
                        metadata['year'] = year
                        all_metadata.append(metadata)
                    
                    if items_path.exists():
                        items = pd.read_csv(items_path)
                        all_items.append(items)
            
            self.metadata_df = pd.concat(all_metadata, ignore_index=True)
            self.items_df = pd.concat(all_items, ignore_index=True)
        
        # Merge metadata with text
        self.df = self.metadata_df.merge(
            self.items_df[['accession', 'item_number', 'item_text']], 
            on=['accession', 'item_number'],
            how='left'
        )
        
        # Clean text
        self.df['item_text'] = self.df['item_text'].fillna('').str[:3000]  # Limit text length
        
        logger.info(f"Loaded {len(self.df)} items from {self.df['year'].nunique()} years")
        logger.info(f"Date range: {self.df['filing_date'].min()} to {self.df['filing_date'].max()}")
        
    def create_contrastive_pairs(self, event_type: str, n_pairs: int = 500) -> List[Dict]:
        """Create contrastive pairs for a specific event type."""
        event_data = self.df[self.df['item_number'] == event_type].copy()
        
        if len(event_data) < 10:
            logger.warning(f"Not enough data for event {event_type}")
            return []
        
        # Get movement threshold for this event
        threshold = EVENT_TYPES[event_type]['threshold']
        
        # Split into high and low performers
        high_performers = event_data[event_data['adjusted_return_pct'] > threshold].copy()
        low_performers = event_data[event_data['adjusted_return_pct'] < -threshold].copy()
        
        logger.info(f"Event {event_type}: {len(high_performers)} high, {len(low_performers)} low performers")
        
        if len(high_performers) < 5 or len(low_performers) < 5:
            logger.warning(f"Not enough contrast for event {event_type}")
            return []
        
        pairs = []
        n_actual_pairs = min(n_pairs, min(len(high_performers), len(low_performers)))
        
        # Sample pairs
        high_sample = high_performers.sample(n=n_actual_pairs, replace=True)
        low_sample = low_performers.sample(n=n_actual_pairs, replace=True)
        
        for (_, high), (_, low) in zip(high_sample.iterrows(), low_sample.iterrows()):
            # Randomly decide which one goes first
            if random.random() < 0.5:
                filing_a, filing_b = high, low
                correct_answer = "A"
            else:
                filing_a, filing_b = low, high
                correct_answer = "B"
            
            # Create the contrastive learning example
            instruction = f"""Compare these two {EVENT_TYPES[event_type]['name']} filings and determine which one led to better 5-day stock performance.

Filing A:
Company: {filing_a['ticker']}
Date: {filing_a['filing_date']}
{filing_a['item_text'][:1500]}

Filing B:
Company: {filing_b['ticker']}
Date: {filing_b['filing_date']}
{filing_b['item_text'][:1500]}

Which filing (A or B) resulted in better stock performance? Explain your reasoning."""

            # Create reasoning based on actual performance
            if correct_answer == "A":
                better_return = filing_a['adjusted_return_pct']
                worse_return = filing_b['adjusted_return_pct']
                better_ticker = filing_a['ticker']
                worse_ticker = filing_b['ticker']
            else:
                better_return = filing_b['adjusted_return_pct']
                worse_return = filing_a['adjusted_return_pct']
                better_ticker = filing_b['ticker']
                worse_ticker = filing_a['ticker']
            
            output = f"""Filing {correct_answer} resulted in better performance.

Analysis:
Filing {correct_answer} ({better_ticker}) shows stronger positive signals with clearer forward guidance and better operational metrics. The language indicates confidence and growth momentum.

Filing {'B' if correct_answer == 'A' else 'A'} ({worse_ticker}) contains more cautious language, uncertainty markers, and potential concerns that likely triggered the negative market reaction.

Confidence: {'HIGH' if abs(better_return - worse_return) > 10 else 'MEDIUM'}
Expected differential: Filing {correct_answer} outperformed by approximately {abs(better_return - worse_return):.1f}%"""

            pairs.append({
                "instruction": instruction,
                "input": "",  # Input is included in instruction for this format
                "output": output,
                "metadata": {
                    "event_type": event_type,
                    "filing_a_return": filing_a['adjusted_return_pct'],
                    "filing_b_return": filing_b['adjusted_return_pct'],
                    "correct_answer": correct_answer,
                    "filing_a_ticker": filing_a['ticker'],
                    "filing_b_ticker": filing_b['ticker'],
                    "filing_a_date": filing_a['filing_date'],
                    "filing_b_date": filing_b['filing_date']
                }
            })
        
        return pairs
    
    def create_single_filing_task(self, event_type: str, n_samples: int = 200) -> List[Dict]:
        """Create single filing prediction tasks for high-signal events."""
        event_data = self.df[self.df['item_number'] == event_type].copy()
        threshold = EVENT_TYPES[event_type]['threshold']
        
        # Filter for high-signal movements only
        high_signal = event_data[abs(event_data['adjusted_return_pct']) > threshold].copy()
        
        if len(high_signal) < 10:
            return []
        
        samples = []
        sample_data = high_signal.sample(n=min(n_samples, len(high_signal)), replace=False)
        
        for _, row in sample_data.iterrows():
            instruction = f"""Analyze this {EVENT_TYPES[event_type]['name']} filing and predict the 5-day market impact.

Company: {row['ticker']}
Filing Date: {row['filing_date']}
Event Type: {event_type} - {EVENT_TYPES[event_type]['name']}

{row['item_text'][:2000]}

Predict the market impact (STRONG_POSITIVE, POSITIVE, NEGATIVE, or STRONG_NEGATIVE) and explain your reasoning."""

            # Determine category based on actual return
            actual_return = row['adjusted_return_pct']
            if actual_return > threshold * 2:
                category = "STRONG_POSITIVE"
                reasoning = "Strong earnings beat, raised guidance, and positive forward indicators"
            elif actual_return > threshold:
                category = "POSITIVE"
                reasoning = "Solid performance with positive momentum indicators"
            elif actual_return < -threshold * 2:
                category = "STRONG_NEGATIVE"
                reasoning = "Significant concerns, missed expectations, or negative guidance"
            else:
                category = "NEGATIVE"
                reasoning = "Disappointing results or uncertainty in forward outlook"
            
            output = f"""Prediction: {category}

Reasoning: {reasoning}

Key Signals:
- Market sentiment indicators suggest {category.lower().replace('_', ' ')} reaction
- Text analysis reveals {'positive' if 'POSITIVE' in category else 'negative'} tone and outlook
- Historical patterns for similar {EVENT_TYPES[event_type]['name']} events support this prediction

Confidence: {'HIGH' if abs(actual_return) > threshold * 1.5 else 'MEDIUM'}
Expected Movement: {abs(actual_return):.1f}% {'increase' if actual_return > 0 else 'decrease'}"""

            samples.append({
                "instruction": instruction,
                "input": "",
                "output": output,
                "metadata": {
                    "event_type": event_type,
                    "actual_return": actual_return,
                    "ticker": row['ticker'],
                    "filing_date": row['filing_date']
                }
            })
        
        return samples
    
    def prepare_dataset(self):
        """Prepare complete dataset with train/val/test splits."""
        all_pairs = []
        all_singles = []
        
        # Create pairs and singles for each event type
        for event_type, info in EVENT_TYPES.items():
            logger.info(f"\nProcessing event {event_type}: {info['name']}")
            
            # Create contrastive pairs
            pairs = self.create_contrastive_pairs(event_type, n_pairs=300)
            all_pairs.extend(pairs)
            
            # Create single filing tasks
            singles = self.create_single_filing_task(event_type, n_samples=200)
            all_singles.extend(singles)
        
        # Combine and shuffle
        all_data = all_pairs + all_singles
        random.shuffle(all_data)
        
        logger.info(f"\nTotal samples created: {len(all_data)}")
        logger.info(f"  - Contrastive pairs: {len(all_pairs)}")
        logger.info(f"  - Single predictions: {len(all_singles)}")
        
        # Split into train/val/test (70/15/15)
        n_total = len(all_data)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]
        
        # Remove metadata from training data (keep for evaluation)
        train_clean = [{k: v for k, v in d.items() if k != 'metadata'} for d in train_data]
        val_clean = [{k: v for k, v in d.items() if k != 'metadata'} for d in val_data]
        test_clean = [{k: v for k, v in d.items() if k != 'metadata'} for d in test_data]
        
        # Save datasets
        with open(self.output_path / "train.json", 'w') as f:
            json.dump(train_clean, f, indent=2)
        
        with open(self.output_path / "val.json", 'w') as f:
            json.dump(val_clean, f, indent=2)
        
        with open(self.output_path / "test.json", 'w') as f:
            json.dump(test_clean, f, indent=2)
        
        # Save metadata separately for evaluation
        with open(self.output_path / "test_metadata.json", 'w') as f:
            json.dump([d['metadata'] for d in test_data], f, indent=2)
        
        # Create dataset info for LlamaFactory
        dataset_info = {
            "contrastive_8k": {
                "file_name": "train.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                }
            },
            "contrastive_8k_val": {
                "file_name": "val.json",
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output"
                }
            }
        }
        
        with open(self.output_path / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info(f"\nDatasets saved:")
        logger.info(f"  - Train: {len(train_data)} samples -> data/train.json")
        logger.info(f"  - Val: {len(val_data)} samples -> data/val.json")
        logger.info(f"  - Test: {len(test_data)} samples -> data/test.json")
        
        # Print statistics
        self.print_statistics(train_data, val_data, test_data)
    
    def print_statistics(self, train, val, test):
        """Print dataset statistics."""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        for name, data in [("Train", train), ("Val", val), ("Test", test)]:
            n_pairs = sum(1 for d in data if "Filing A:" in d['instruction'])
            n_singles = len(data) - n_pairs
            
            print(f"\n{name} Set:")
            print(f"  Total: {len(data)} samples")
            print(f"  Contrastive pairs: {n_pairs}")
            print(f"  Single predictions: {n_singles}")
            
            if data and 'metadata' in data[0]:
                event_dist = {}
                for d in data:
                    event = d['metadata'].get('event_type', 'unknown')
                    event_dist[event] = event_dist.get(event, 0) + 1
                
                print(f"  Event distribution:")
                for event, count in sorted(event_dist.items()):
                    if event in EVENT_TYPES:
                        print(f"    {event} ({EVENT_TYPES[event]['name']}): {count}")

def main():
    """Main execution."""
    logger.info("Starting contrastive data preparation...")
    
    preparator = ContrastiveDataPreparator()
    preparator.prepare_dataset()
    
    logger.info("\n✅ Data preparation complete!")
    logger.info("\nNext steps:")
    logger.info("1. Copy data/ folder to your GPU server")
    logger.info("2. Run training with train_contrastive.sh")
    logger.info("3. Evaluate with inference_contrastive.py")

if __name__ == "__main__":
    main()