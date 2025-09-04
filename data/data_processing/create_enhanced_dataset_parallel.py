#!/usr/bin/env python3
"""
Create Enhanced Dataset with Parallel Processing
=================================================
Processes 6 years in parallel for maximum efficiency
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import os
import sys

# Import the main creator class
sys.path.append(str(Path(__file__).parent))
from create_enhanced_dataset_v2 import EnhancedDatasetCreator

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_single_year(year):
    """Process a single year - function for parallel execution"""
    try:
        logger.info(f"Starting processing for year {year}")
        
        # Create an instance for this year only
        creator = EnhancedDatasetCreator(years=[year])
        
        # Process the year
        year_df = creator.process_year(year)
        
        if year_df is not None:
            logger.info(f"✓ Completed year {year}: {len(year_df)} filings processed")
            return year_df
        else:
            logger.warning(f"✗ No data for year {year}")
            return None
            
    except Exception as e:
        logger.error(f"✗ Error processing year {year}: {e}")
        return None


def main():
    """Main parallel processing function"""
    parser = argparse.ArgumentParser(description='Create enhanced dataset with parallel processing')
    parser.add_argument('--years', type=int, nargs='+', 
                       default=[2014, 2015, 2016, 2017, 2018, 2019],
                       help='Years to process')
    parser.add_argument('--workers', type=int, default=6,
                       help='Number of parallel workers (default: 6)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("ENHANCED DATASET CREATION - PARALLEL VERSION")
    logger.info("="*60)
    logger.info(f"Years to process: {args.years}")
    logger.info(f"Parallel workers: {args.workers}")
    
    # Set HuggingFace token
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')
    if hf_token:
        logger.info("✓ HuggingFace token found")
    
    # Process years in parallel
    all_dfs = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all years for processing
        future_to_year = {executor.submit(process_single_year, year): year 
                         for year in args.years}
        
        # Collect results as they complete
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                year_df = future.result()
                if year_df is not None:
                    all_dfs.append(year_df)
            except Exception as e:
                logger.error(f"Failed to get result for year {year}: {e}")
    
    # Combine all years
    if all_dfs:
        logger.info("\n" + "="*60)
        logger.info("COMBINING ALL YEARS")
        logger.info("="*60)
        
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Sort by date
        full_df = full_df.sort_values('filing_date').reset_index(drop=True)
        
        # Save final dataset
        output_dir = Path("enhanced_eight_k")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"enhanced_dataset_{min(args.years)}_{max(args.years)}.csv"
        full_df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Saved enhanced dataset to {output_file}")
        logger.info(f"  Total filings: {len(full_df)}")
        logger.info(f"  Years: {sorted(full_df['year'].unique())}")
        
        # Save summary statistics
        summary = {
            'total_filings': len(full_df),
            'years': list(full_df['year'].unique()),
            'data_structure': {
                'main_csv': str(output_file.name),
                'text_files': 'texts/merged_texts_YYYY.json',
                'embeddings': 'embeddings/YYYY/embeddings.npy',
                'accession_order': 'embeddings/YYYY/accession_order.json'
            },
            'features': {
                'metadata': ['ticker', 'filing_date', 'accession', 'year'],
                'items': ['items_present', 'item_count', 'text_length'],
                'company': ['sector', 'industry'],
                'market': ['vix_level'],
                'momentum': ['momentum_7d', 'momentum_30d', 'momentum_90d', 'momentum_365d'],
                'target': ['signal', 'adjusted_return_pct', 'outperformed_market'],
                'phase1_embedding': ['embedding_index', 'embedding_year']
            },
            'signal_distribution': full_df['signal'].value_counts().to_dict(),
            'sector_distribution': full_df['sector'].value_counts().head(10).to_dict(),
            'items_statistics': {
                'avg_items_per_filing': full_df['item_count'].mean(),
                'max_items': full_df['item_count'].max(),
                'min_items': full_df['item_count'].min(),
            },
            'text_statistics': {
                'avg_text_length': full_df['text_length'].mean(),
                'max_text_length': full_df['text_length'].max(),
                'min_text_length': full_df['text_length'].min()
            },
            'momentum_coverage': {
                '7d': full_df['momentum_7d'].notna().sum() / len(full_df),
                '30d': full_df['momentum_30d'].notna().sum() / len(full_df),
                '90d': full_df['momentum_90d'].notna().sum() / len(full_df),
                '365d': full_df['momentum_365d'].notna().sum() / len(full_df)
            },
            'vix_coverage': full_df['vix_level'].notna().sum() / len(full_df),
            'sector_coverage': full_df['sector'].notna().sum() / len(full_df)
        }
        
        summary_file = output_dir / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✓ Saved summary to {summary_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"Signal distribution:")
        for signal, count in summary['signal_distribution'].items():
            logger.info(f"  {signal}: {count} ({count/len(full_df)*100:.1f}%)")
        logger.info(f"VIX coverage: {summary['vix_coverage']*100:.1f}%")
        logger.info(f"Sector coverage: {summary['sector_coverage']*100:.1f}%")
        logger.info(f"Momentum coverage (7d): {summary['momentum_coverage']['7d']*100:.1f}%")
        
        return full_df
    else:
        logger.error("No data processed successfully!")
        return None


if __name__ == "__main__":
    main()