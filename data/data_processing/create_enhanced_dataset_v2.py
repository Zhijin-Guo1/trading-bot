#!/usr/bin/env python3
"""
Create Enhanced Dataset with All Features - Version 2
======================================================
Improvements:
1. Saves merged_text as separate JSON file
2. Better tracking of item extraction
"""

import pandas as pd
import numpy as np
import json
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedDatasetCreator:
    """Create enhanced dataset with all features"""
    
    def __init__(self, years=None):
        self.years = years or [2014, 2015, 2016, 2017, 2018, 2019]
        self.processed_dir = Path("processed_data")
        self.output_dir = Path("enhanced_eight_k")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create embeddings directory
        self.embeddings_dir = self.output_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Create texts directory for saving merged texts
        self.texts_dir = self.output_dir / "texts"
        self.texts_dir.mkdir(exist_ok=True)
        
        # Initialize models and caches
        self.embedding_model = None
        self.vix_cache = {}
        self.price_cache = {}
        self.spy_cache = {}
        
    def load_processed_items(self, year):
        """Load processed items for a year"""
        items_file = self.processed_dir / str(year) / "items_separated.csv"
        if not items_file.exists():
            logger.warning(f"No processed data for {year}")
            return None
        
        df = pd.read_csv(items_file)
        logger.info(f"Loaded {len(df)} items for {year}")
        return df
    
    def merge_items_by_filing(self, df, year):
        """Merge items back to filing level and save texts separately"""
        logger.info("Merging items by filing...")
        
        merged_data = []
        merged_texts = {}  # Store texts separately
        
        # Group by accession (unique filing ID)
        for accession, group in tqdm(df.groupby('accession'), desc="Merging filings"):
            # Get filing metadata from first row
            filing_info = {
                'ticker': group.iloc[0]['ticker'],
                'filing_date': group.iloc[0]['filing_date'],
                'accession': accession,
                'adjusted_return_pct': group.iloc[0]['adjusted_return_pct'],
                'outperformed_market': group.iloc[0]['outperformed_market']
            }
            
            # Get list of items present with descriptions
            items_list = []
            for _, row in group.iterrows():
                item_num = row['item_number']
                if item_num != 'MIXED':
                    items_list.append(item_num)
            
            # Sort and deduplicate items
            items_list = sorted(list(set(items_list)))
            filing_info['items_present'] = ','.join(items_list) if items_list else 'MIXED'
            filing_info['item_count'] = len(items_list) if items_list else 1
            
            # Concatenate all item texts with item headers
            text_parts = []
            for _, row in group.iterrows():
                if pd.notna(row['item_text']):
                    # Add item header for clarity
                    item_header = f"\n[ITEM {row['item_number']}: {row['item_description']}]\n"
                    text_parts.append(item_header + row['item_text'])
            
            merged_text = '\n\n'.join(text_parts)
            filing_info['text_length'] = len(merged_text)
            
            # Store text separately
            merged_texts[accession] = {
                'ticker': filing_info['ticker'],
                'filing_date': filing_info['filing_date'],
                'text': merged_text,
                'items': items_list
            }
            
            merged_data.append(filing_info)
        
        # Save merged texts to JSON
        texts_file = self.texts_dir / f"merged_texts_{year}.json"
        with open(texts_file, 'w') as f:
            json.dump(merged_texts, f, indent=2)
        logger.info(f"Saved merged texts to {texts_file}")
        
        merged_df = pd.DataFrame(merged_data)
        logger.info(f"Merged to {len(merged_df)} filings")
        return merged_df, merged_texts
    
    def add_sector_information(self, df):
        """Add sector and industry from SP500 companies"""
        logger.info("Adding sector information...")
        
        # Load SP500 companies data
        sp500_file = Path("../data/company_data/sp500_companies.csv")
        if not sp500_file.exists():
            logger.warning("SP500 companies file not found, skipping sector info")
            df['sector'] = None
            df['industry'] = None
            return df
        
        sp500_df = pd.read_csv(sp500_file)
        
        # Create mapping dictionary
        sector_map = {}
        industry_map = {}
        for _, row in sp500_df.iterrows():
            ticker = row['symbol']
            sector_map[ticker] = row['sector']
            industry_map[ticker] = row['industry']
        
        # Map sectors and industries
        df['sector'] = df['ticker'].map(sector_map)
        df['industry'] = df['ticker'].map(industry_map)
        
        logger.info(f"Mapped sectors for {df['sector'].notna().sum()}/{len(df)} filings")
        return df
    
    def calculate_momentum_features(self, df):
        """Calculate historical momentum (market-adjusted)"""
        logger.info("Calculating momentum features...")
        
        momentum_features = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Calculating momentum"):
            ticker = row['ticker']
            filing_date = pd.to_datetime(row['filing_date'])
            
            features = {
                'momentum_7d': None,
                'momentum_30d': None,
                'momentum_90d': None,
                'momentum_365d': None
            }
            
            # Define lookback periods
            periods = {
                'momentum_7d': 7,
                'momentum_30d': 30,
                'momentum_90d': 90,
                'momentum_365d': 365
            }
            
            for feature_name, days in periods.items():
                try:
                    # Get stock prices
                    start = filing_date - timedelta(days=days+10)
                    end = filing_date
                    
                    # Check cache first
                    cache_key = f"{ticker}_{start.date()}_{end.date()}"
                    if cache_key not in self.price_cache:
                        stock = yf.Ticker(ticker)
                        stock_hist = stock.history(start=start, end=end)
                        self.price_cache[cache_key] = stock_hist
                    else:
                        stock_hist = self.price_cache[cache_key]
                    
                    # Get SPY prices
                    spy_key = f"SPY_{start.date()}_{end.date()}"
                    if spy_key not in self.spy_cache:
                        spy = yf.Ticker("SPY")
                        spy_hist = spy.history(start=start, end=end)
                        self.spy_cache[spy_key] = spy_hist
                    else:
                        spy_hist = self.spy_cache[spy_key]
                    
                    if len(stock_hist) >= 2 and len(spy_hist) >= 2:
                        # Calculate returns
                        stock_return = ((stock_hist['Close'].iloc[-1] - stock_hist['Close'].iloc[0]) / 
                                      stock_hist['Close'].iloc[0] * 100)
                        spy_return = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[0]) / 
                                    spy_hist['Close'].iloc[0] * 100)
                        
                        features[feature_name] = stock_return - spy_return
                except:
                    pass
            
            momentum_features.append(features)
        
        # Add features to dataframe
        momentum_df = pd.DataFrame(momentum_features)
        for col in momentum_df.columns:
            df[col] = momentum_df[col]
        
        logger.info(f"Added momentum features for {df['momentum_7d'].notna().sum()}/{len(df)} filings")
        return df
    
    def add_vix_volatility(self, df):
        """Add VIX volatility level for each filing date"""
        logger.info("Adding VIX volatility...")
        
        vix_values = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching VIX"):
            filing_date = row['filing_date']
            
            if filing_date in self.vix_cache:
                vix_values.append(self.vix_cache[filing_date])
                continue
            
            try:
                filing_dt = pd.to_datetime(filing_date)
                vix = yf.Ticker("^VIX")
                # IMPORTANT: Only look at VIX data BEFORE or ON filing date (no future data)
                vix_hist = vix.history(start=filing_dt - timedelta(days=10),
                                      end=filing_dt + timedelta(days=1))  # +1 to include filing date
                
                if not vix_hist.empty:
                    # Filter to only dates <= filing date to avoid look-ahead bias
                    vix_hist = vix_hist[vix_hist.index.date <= filing_dt.date()]
                    
                    if not vix_hist.empty:
                        # Get the most recent VIX value available at filing date
                        vix_value = vix_hist.iloc[-1]['Close']  # Last available value
                        self.vix_cache[filing_date] = vix_value
                        vix_values.append(vix_value)
                    else:
                        vix_values.append(None)
                else:
                    vix_values.append(None)
            except:
                vix_values.append(None)
        
        df['vix_level'] = vix_values
        logger.info(f"Added VIX for {df['vix_level'].notna().sum()}/{len(df)} filings")
        return df
    
    def create_3class_signal(self, df):
        """Create 3-class signal: UP (>1%), DOWN (<-1%), STAY (-1% to 1%)"""
        logger.info("Creating 3-class signal...")
        
        def classify_return(ret):
            if pd.isna(ret):
                return None
            elif ret > 1.0:
                return 'UP'
            elif ret < -1.0:
                return 'DOWN'
            else:
                return 'STAY'
        
        df['signal'] = df['adjusted_return_pct'].apply(classify_return)
        
        # Print distribution
        signal_dist = df['signal'].value_counts()
        logger.info("Signal distribution:")
        for signal, count in signal_dist.items():
            logger.info(f"  {signal}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def generate_embeddings(self, merged_texts, year):
        """Generate document embeddings for merged texts"""
        logger.info(f"Generating embeddings for {year}...")
        
        if self.embedding_model is None:
            logger.info("Loading phase1_embedding model (all-MiniLM-L6-v2)...")
            # Use HuggingFace token if available
            hf_token = os.environ.get('HUGGINGFACE_TOKEN', None)
            if hf_token:
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                                          use_auth_token=hf_token)
            else:
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Extract texts in order of accessions
        accessions = sorted(merged_texts.keys())
        texts = [merged_texts[acc]['text'] for acc in accessions]
        
        # Generate embeddings in batches
        embeddings = []
        batch_size = 32
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {year}"):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        all_embeddings = np.vstack(embeddings)
        
        # Save embeddings
        embeddings_file = self.embeddings_dir / str(year) / "embeddings.npy"
        embeddings_file.parent.mkdir(exist_ok=True)
        np.save(embeddings_file, all_embeddings)
        
        # Also save accession order for reference
        accession_file = self.embeddings_dir / str(year) / "accession_order.json"
        with open(accession_file, 'w') as f:
            json.dump(accessions, f)
        
        logger.info(f"Saved embeddings with shape {all_embeddings.shape} to {embeddings_file}")
        return accessions
    
    def process_year(self, year):
        """Process a single year with all features"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing year {year}")
        logger.info(f"{'='*60}")
        
        # Load processed items
        items_df = self.load_processed_items(year)
        if items_df is None:
            return None
        
        # Merge items by filing and save texts
        merged_df, merged_texts = self.merge_items_by_filing(items_df, year)
        
        # Add all features
        merged_df = self.add_sector_information(merged_df)
        merged_df = self.calculate_momentum_features(merged_df)
        merged_df = self.add_vix_volatility(merged_df)
        merged_df = self.create_3class_signal(merged_df)
        
        # Generate embeddings
        accession_order = self.generate_embeddings(merged_texts, year)
        
        # Add phase1_embedding index based on accession order
        embedding_map = {acc: idx for idx, acc in enumerate(accession_order)}
        merged_df['embedding_index'] = merged_df['accession'].map(embedding_map)
        
        # Add year column
        merged_df['year'] = year
        merged_df['embedding_year'] = year
        
        return merged_df
    
    def process_year_wrapper(self, year):
        """Wrapper for parallel processing of years"""
        # Recreate the instance to avoid pickle issues
        creator = EnhancedDatasetCreator(years=[year])
        return creator.process_year(year)
    
    def create_full_dataset(self):
        """Process all years and create final dataset"""
        logger.info("="*60)
        logger.info("CREATING ENHANCED DATASET V2")
        logger.info("="*60)
        logger.info(f"Processing {len(self.years)} years in parallel with 6 workers")
        
        all_dfs = []
        
        # Process years in parallel
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=6) as executor:
            # Submit all years for processing
            future_to_year = {executor.submit(self.process_year_wrapper, year): year 
                             for year in self.years}
            
            # Collect results as they complete
            for future in as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    year_df = future.result()
                    if year_df is not None:
                        all_dfs.append(year_df)
                        logger.info(f"✓ Completed processing year {year}")
                except Exception as e:
                    logger.error(f"✗ Error processing year {year}: {e}")
        
        if not all_dfs:
            logger.error("No data processed!")
            return
        
        # Combine all years
        full_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save final dataset (without text, which is in separate files)
        min_year = min(self.years)
        max_year = max(self.years)
        output_file = self.output_dir / f"enhanced_dataset_{min_year}_{max_year}.csv"
        full_df.to_csv(output_file, index=False)
        logger.info(f"Saved enhanced dataset to {output_file}")
        
        # Save summary statistics
        self.save_summary(full_df)
        
        return full_df
    
    def save_summary(self, df):
        """Save summary statistics"""
        summary = {
            'total_filings': len(df),
            'years': list(df['year'].unique()),
            'data_structure': {
                'main_csv': 'enhanced_dataset_2014_2019.csv',
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
            'signal_distribution': df['signal'].value_counts().to_dict(),
            'sector_distribution': df['sector'].value_counts().head(10).to_dict(),
            'items_statistics': {
                'avg_items_per_filing': df['item_count'].mean(),
                'max_items': df['item_count'].max(),
                'min_items': df['item_count'].min(),
                'most_common_items': df['items_present'].value_counts().head(10).to_dict()
            },
            'text_statistics': {
                'avg_text_length': df['text_length'].mean(),
                'max_text_length': df['text_length'].max(),
                'min_text_length': df['text_length'].min()
            },
            'momentum_coverage': {
                '7d': df['momentum_7d'].notna().sum() / len(df),
                '30d': df['momentum_30d'].notna().sum() / len(df),
                '90d': df['momentum_90d'].notna().sum() / len(df),
                '365d': df['momentum_365d'].notna().sum() / len(df)
            },
            'vix_coverage': df['vix_level'].notna().sum() / len(df),
            'sector_coverage': df['sector'].notna().sum() / len(df)
        }
        
        summary_file = self.output_dir / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved summary to {summary_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"Total filings: {summary['total_filings']}")
        logger.info(f"Years: {summary['years']}")
        logger.info(f"\nData Structure:")
        logger.info(f"  Main CSV: enhanced_eight_k/{summary['data_structure']['main_csv']}")
        logger.info(f"  Text files: enhanced_eight_k/{summary['data_structure']['text_files']}")
        logger.info(f"  Embeddings: enhanced_eight_k/{summary['data_structure']['embeddings']}")
        logger.info(f"\nSignal distribution:")
        for signal, count in summary['signal_distribution'].items():
            logger.info(f"  {signal}: {count}")
        logger.info(f"\nAverage items per filing: {summary['items_statistics']['avg_items_per_filing']:.2f}")
        logger.info(f"Average text length: {summary['text_statistics']['avg_text_length']:.0f} chars")
        logger.info(f"VIX coverage: {summary['vix_coverage']*100:.1f}%")
        logger.info(f"Sector coverage: {summary['sector_coverage']*100:.1f}%")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create enhanced dataset with all features (V2)')
    parser.add_argument('--years', nargs='+', type=int,
                       default=[2014, 2015, 2016, 2017, 2018, 2019],
                       help='Years to process (default: 2014-2019)')
    
    args = parser.parse_args()
    
    # Check for HuggingFace token in environment
    if 'HUGGINGFACE_TOKEN' not in os.environ:
        logger.warning("HUGGINGFACE_TOKEN not found in environment variables.")
        logger.warning("Please set it using: export HUGGINGFACE_TOKEN='your_token_here'")
        logger.warning("Or create a .env file with: HUGGINGFACE_TOKEN=your_token_here")
        logger.info("Continuing without HuggingFace token - some features may not work.")
    
    # Create enhanced dataset
    creator = EnhancedDatasetCreator(years=args.years)
    creator.create_full_dataset()
    
    min_year = min(args.years)
    max_year = max(args.years)
    logger.info("\n✓ Enhanced dataset creation complete!")
    logger.info("\nData saved in:")
    logger.info(f"  - enhanced_eight_k/enhanced_dataset_{min_year}_{max_year}.csv (features)")
    logger.info("  - enhanced_eight_k/texts/merged_texts_YYYY.json (full texts)")
    logger.info("  - enhanced_eight_k/embeddings/YYYY/embeddings.npy (embeddings)")


if __name__ == "__main__":
    main()