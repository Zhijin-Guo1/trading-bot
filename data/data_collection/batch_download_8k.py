"""
Batch download 8-K filings with resumable progress
Optimized for large-scale data collection with checkpointing
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import re
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchEightKCollector:
    def __init__(self, start_year: int = 2014, end_year: int = 2024, batch_size: int = 10):
        """Initialize batch 8-K collector"""
        self.start_year = start_year
        self.end_year = end_year
        self.batch_size = batch_size
        self.base_url = "https://data.sec.gov"
        self.archives_url = "https://www.sec.gov/Archives/edgar/data"
        self.headers = {
            "User-Agent": "Academic Research Bot research@university.edu"
        }
        
        # Create directory structure
        self.base_dir = Path("data_collection/eight_k")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Create year directories
        for year in range(start_year, end_year + 1):
            (self.base_dir / str(year)).mkdir(exist_ok=True)
        
        # Load S&P 500 companies
        self.sp500_df = self.load_sp500_companies()
        
        # Get CIK mapping
        self.ticker_to_cik = self.get_cik_mapping()
        
        # Load checkpoint if exists
        self.progress = self.load_checkpoint()
        
    def load_sp500_companies(self) -> pd.DataFrame:
        """Load S&P 500 companies list"""
        analysis_path = Path("analysis/data/sp500/sp500_companies.csv")
        local_path = self.base_dir / "sp500_companies.csv"
        
        if analysis_path.exists():
            df = pd.read_csv(analysis_path)
            df.to_csv(local_path, index=False)
            logger.info(f"Loaded {len(df)} S&P 500 companies")
            return df
        elif local_path.exists():
            df = pd.read_csv(local_path)
            logger.info(f"Loaded {len(df)} S&P 500 companies")
            return df
        else:
            logger.info("Downloading S&P 500 list from Wikipedia")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0][['Symbol', 'Security', 'GICS Sector']]
            df['Symbol'] = df['Symbol'].str.replace('.', '-')
            df.to_csv(local_path, index=False)
            logger.info(f"Downloaded {len(df)} S&P 500 companies")
            return df
    
    def get_cik_mapping(self) -> Dict[str, str]:
        """Get ticker to CIK mapping"""
        cache_path = self.base_dir / "cik_mapping.json"
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            data = response.json()
            
            ticker_to_cik = {}
            for item in data.values():
                ticker = item['ticker']
                cik = str(item['cik_str']).zfill(10)
                ticker_to_cik[ticker] = cik
            
            with open(cache_path, 'w') as f:
                json.dump(ticker_to_cik, f)
            
            logger.info(f"Cached CIK mappings for {len(ticker_to_cik)} companies")
            return ticker_to_cik
        except Exception as e:
            logger.error(f"Error loading CIK mapping: {e}")
            return {}
    
    def load_checkpoint(self) -> Dict:
        """Load progress checkpoint"""
        checkpoint_file = self.checkpoint_dir / "progress.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                progress = json.load(f)
                logger.info(f"Resumed from checkpoint: {progress['completed_count']} items completed")
                return progress
        return {
            'completed_years': [],
            'completed_companies': {},
            'completed_count': 0
        }
    
    def save_checkpoint(self):
        """Save progress checkpoint"""
        checkpoint_file = self.checkpoint_dir / "progress.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def extract_8k_text(self, content: str) -> str:
        """Extract relevant text from 8-K filing"""
        text_parts = []
        
        # Look for specific item patterns in 8-K
        important_items = [
            r'(?i)item\s*1\.01.*?entry.*?material.*?agreement',
            r'(?i)item\s*2\.01.*?completion.*?acquisition',
            r'(?i)item\s*2\.02.*?results.*?operations',
            r'(?i)item\s*2\.03.*?creation.*?obligation',
            r'(?i)item\s*5\.02.*?departure.*?directors',
            r'(?i)item\s*7\.01.*?regulation.*?fd',
            r'(?i)item\s*8\.01.*?other.*?events'
        ]
        
        for pattern in important_items:
            matches = re.findall(pattern + r'.*?(?=item\s*[0-9]|signature|$)', 
                               content, re.DOTALL)
            text_parts.extend(matches[:2])  # Limit each item
        
        # If no specific items, get general content
        if not text_parts:
            items_pattern = r'(?i)item\s*[0-9.]+.*?(?=item\s*[0-9]|signature|$)'
            items = re.findall(items_pattern, content, re.DOTALL)
            text_parts = items[:5]
        
        # Clean and combine
        text = ' '.join(text_parts)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\%\$]', ' ', text)
        
        return text[:50000].strip()
    
    def get_prices_batch(self, ticker: str, dates: List[str]) -> Dict[str, Dict]:
        """Get prices for multiple dates in batch"""
        if not dates:
            return {}
        
        try:
            # Get date range
            min_date = min(datetime.strptime(d, "%Y-%m-%d") for d in dates)
            max_date = max(datetime.strptime(d, "%Y-%m-%d") for d in dates)
            
            # Extend range for T+5 calculations
            start_date = min_date - timedelta(days=5)
            end_date = max_date + timedelta(days=15)
            
            # Get stock and SPY data
            stock = yf.Ticker(ticker)
            stock_hist = stock.history(start=start_date, end=end_date, interval="1d")
            
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(start=start_date, end=end_date, interval="1d")
            
            results = {}
            
            for filing_date in dates:
                filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
                
                # Find T0 and T5
                t0_idx = None
                t5_idx = None
                
                # Find next trading day after filing
                for i in range(len(stock_hist)):
                    if stock_hist.index[i].date() > filing_dt.date():
                        t0_idx = i
                        break
                
                if t0_idx is not None and t0_idx + 4 < len(stock_hist):
                    t5_idx = t0_idx + 4
                    
                    # Get corresponding SPY indices
                    t0_date = stock_hist.index[t0_idx].date()
                    t5_date = stock_hist.index[t5_idx].date()
                    
                    spy_t0_idx = spy_hist.index.get_indexer([t0_date], method='nearest')[0]
                    spy_t5_idx = spy_hist.index.get_indexer([t5_date], method='nearest')[0]
                    
                    if spy_t0_idx < len(spy_hist) and spy_t5_idx < len(spy_hist):
                        # Calculate returns
                        stock_p0 = stock_hist.iloc[t0_idx]['Close']
                        stock_p5 = stock_hist.iloc[t5_idx]['Close']
                        spy_p0 = spy_hist.iloc[spy_t0_idx]['Close']
                        spy_p5 = spy_hist.iloc[spy_t5_idx]['Close']
                        
                        stock_ret = ((stock_p5 - stock_p0) / stock_p0) * 100
                        spy_ret = ((spy_p5 - spy_p0) / spy_p0) * 100
                        adj_ret = stock_ret - spy_ret
                        
                        results[filing_date] = {
                            'stock_price_t0': round(stock_p0, 2),
                            'stock_price_t5': round(stock_p5, 2),
                            'spy_price_t0': round(spy_p0, 2),
                            'spy_price_t5': round(spy_p5, 2),
                            'stock_return_pct': round(stock_ret, 3),
                            'market_return_pct': round(spy_ret, 3),
                            'adjusted_return_pct': round(adj_ret, 3),
                            'outperformed_market': 1 if adj_ret > 0 else 0,
                            't0_date': str(t0_date),
                            't5_date': str(t5_date)
                        }
            
            return results
            
        except Exception as e:
            logger.debug(f"Error getting batch prices for {ticker}: {e}")
            return {}
    
    def process_company_year(self, ticker: str, year: int) -> List[Dict]:
        """Process all 8-K filings for a company in a year"""
        # Check if already processed
        if year in self.progress.get('completed_companies', {}).get(ticker, []):
            return []
        
        cik = self.ticker_to_cik.get(ticker)
        if not cik:
            return []
        
        session = requests.Session()
        session.headers.update(self.headers)
        
        try:
            # Get company submissions
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            filings = []
            filing_dates = []
            
            # Parse filings
            recent = data.get('filings', {}).get('recent', {})
            if recent:
                forms = recent.get('form', [])
                dates = recent.get('filingDate', [])
                accessions = recent.get('accessionNumber', [])
                
                # Collect all 8-K filings for the year
                year_filings = []
                for i in range(len(forms)):
                    if dates[i].startswith(str(year)) and forms[i] == '8-K':
                        year_filings.append({
                            'date': dates[i],
                            'accession': accessions[i]
                        })
                        filing_dates.append(dates[i])
                
                # Get batch prices
                price_data = self.get_prices_batch(ticker, filing_dates)
                
                # Process each filing
                for filing_info in year_filings:
                    # Get text
                    accession = filing_info['accession']
                    accession_clean = accession.replace('-', '')
                    filing_url = f"{self.archives_url}/{cik.lstrip('0')}/{accession_clean}/{accession}.txt"
                    
                    try:
                        filing_response = session.get(filing_url, timeout=30)
                        filing_response.raise_for_status()
                        text = self.extract_8k_text(filing_response.text)
                    except:
                        text = ""
                    
                    # Combine with price data
                    filing_data = {
                        'ticker': ticker,
                        'cik': cik,
                        'filing_date': filing_info['date'],
                        'accession': accession,
                        'text': text,
                        'text_length': len(text)
                    }
                    
                    # Add price data if available
                    if filing_info['date'] in price_data:
                        filing_data.update(price_data[filing_info['date']])
                    
                    filings.append(filing_data)
            
            # Update progress
            if ticker not in self.progress['completed_companies']:
                self.progress['completed_companies'][ticker] = []
            self.progress['completed_companies'][ticker].append(year)
            self.progress['completed_count'] += len(filings)
            
            return filings
            
        except Exception as e:
            logger.debug(f"Error processing {ticker} for {year}: {e}")
            return []
    
    def process_year_batch(self, year: int):
        """Process all companies for a year in batches"""
        if year in self.progress['completed_years']:
            logger.info(f"Year {year} already completed, skipping")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Year {year}")
        logger.info(f"{'='*60}")
        
        all_filings = []
        companies = self.sp500_df['Symbol'].tolist()
        
        # Process in batches
        for i in range(0, len(companies), self.batch_size):
            batch = companies[i:i+self.batch_size]
            batch_filings = []
            
            # Process batch with progress bar
            with tqdm(total=len(batch), desc=f"Batch {i//self.batch_size + 1}") as pbar:
                for ticker in batch:
                    filings = self.process_company_year(ticker, year)
                    batch_filings.extend(filings)
                    pbar.update(1)
                    time.sleep(0.1)  # Rate limiting
            
            all_filings.extend(batch_filings)
            
            # Save intermediate results
            if batch_filings:
                self.save_year_data(year, all_filings, intermediate=True)
            
            # Save checkpoint
            self.save_checkpoint()
        
        # Final save for the year
        if all_filings:
            self.save_year_data(year, all_filings, intermediate=False)
            self.progress['completed_years'].append(year)
            self.save_checkpoint()
            
            logger.info(f"Year {year} complete: {len(all_filings)} filings")
    
    def save_year_data(self, year: int, filings: List[Dict], intermediate: bool = False):
        """Save year data to files"""
        if not filings:
            return
        
        df = pd.DataFrame(filings)
        year_dir = self.base_dir / str(year)
        
        # Save metadata
        suffix = "_intermediate" if intermediate else ""
        metadata_file = year_dir / f"8K_{year}_metadata{suffix}.csv"
        df_meta = df.drop('text', axis=1, errors='ignore')
        df_meta.to_csv(metadata_file, index=False)
        
        # Save text
        text_file = year_dir / f"8K_{year}_text{suffix}.json"
        if 'text' in df.columns:
            text_data = df[['ticker', 'filing_date', 'accession', 'text']].to_dict('records')
            with open(text_file, 'w') as f:
                json.dump(text_data, f)
        
        # Save complete data with returns
        if 'stock_price_t0' in df.columns:
            complete_file = year_dir / f"8K_{year}_complete{suffix}.csv"
            df_complete = df[df['stock_price_t0'].notna()]
            df_complete = df_complete.drop('text', axis=1, errors='ignore')
            df_complete.to_csv(complete_file, index=False)
    
    def run(self):
        """Run the batch download process"""
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH 8-K DOWNLOAD: {self.start_year}-{self.end_year}")
        logger.info(f"{'='*60}")
        logger.info(f"Companies: {len(self.sp500_df)}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Progress: {self.progress['completed_count']} filings completed")
        logger.info(f"{'='*60}\n")
        
        # Process each year
        for year in range(self.start_year, self.end_year + 1):
            self.process_year_batch(year)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("BATCH DOWNLOAD COMPLETE")
        print("="*60)
        
        total_filings = 0
        for year in range(self.start_year, self.end_year + 1):
            complete_file = self.base_dir / str(year) / f"8K_{year}_complete.csv"
            if complete_file.exists():
                df = pd.read_csv(complete_file)
                total_filings += len(df)
                print(f"Year {year}: {len(df):,} filings")
        
        print(f"\nTotal filings downloaded: {total_filings:,}")
        print(f"Data location: {self.base_dir}")
        print("="*60)

def main():
    """Main function"""
    collector = BatchEightKCollector(
        start_year=2014,
        end_year=2024,
        batch_size=10  # Process 10 companies at a time
    )
    collector.run()

if __name__ == "__main__":
    main()