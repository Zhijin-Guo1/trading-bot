"""
Process 8-K Filings with Item Separation (2021-2024)
====================================================
Creates items_separated.csv files for each year with proper item separation.
No sampling - processes ALL filings.
"""

import pandas as pd
import numpy as np
import json
import re
import requests
import time
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EightKItemProcessor:
    def __init__(self, years=None):
        """Initialize processor for specified years."""
        self.years = years or [2021, 2022, 2023, 2024]
        self.base_dir = Path("processed_data")
        self.base_dir.mkdir(exist_ok=True)
        
        # Item descriptions mapping
        self.item_descriptions = {
            '1.01': 'Entry into Material Agreement',
            '1.02': 'Termination of Material Agreement',
            '1.03': 'Bankruptcy or Receivership',
            '1.04': 'Mine Safety',
            '2.01': 'Completion of Acquisition/Disposition',
            '2.02': 'Results of Operations',
            '2.03': 'Creation of Financial Obligation',
            '2.04': 'Triggering Events',
            '2.05': 'Exit/Restructuring Costs',
            '2.06': 'Material Impairments',
            '3.01': 'Delisting Notice',
            '3.02': 'Unregistered Securities Sales',
            '3.03': 'Material Modifications',
            '4.01': 'Accountant Changes',
            '4.02': 'Non-Reliance on Financials',
            '5.01': 'Control Changes',
            '5.02': 'Officer/Director Changes',
            '5.03': 'Bylaw Amendments',
            '5.04': 'Trading Suspension',
            '5.05': 'Code of Ethics Changes',
            '5.06': 'Shell Company Status Change',
            '5.07': 'Shareholder Matters',
            '5.08': 'Shareholder Nominations',
            '6.01': 'ABS Information',
            '6.02': 'Change of Servicer',
            '6.03': 'Change in Credit Enhancement',
            '7.01': 'Regulation FD Disclosure',
            '8.01': 'Other Material Events',
            '9.01': 'Financial Statements/Exhibits'
        }
        
        self.headers = {
            'User-Agent': 'Academic Research Bot (research@university.edu)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
    
    def load_year_data(self, year):
        """Load existing 8-K data for a year."""
        data_dir = Path(f"eight_k/{year}")
        
        # Load returns data
        complete_file = data_dir / f"8K_{year}_complete.csv"
        if not complete_file.exists():
            logger.error(f"No data found for {year} at {complete_file}")
            return None, None
        
        df_returns = pd.read_csv(complete_file)
        
        # Load text data
        text_file = data_dir / f"8K_{year}_text.json"
        if not text_file.exists():
            logger.error(f"No text data found for {year}")
            return df_returns, {}
        
        with open(text_file, 'r') as f:
            text_list = json.load(f)
        
        # Convert to dict keyed by accession
        text_dict = {}
        for item in text_list:
            if isinstance(item, dict) and 'accession' in item:
                text_dict[item['accession']] = item.get('text', '')
        
        logger.info(f"Loaded {len(df_returns)} filings for {year}")
        return df_returns, text_dict
    
    def download_filing(self, cik, accession):
        """Download full filing text from SEC."""
        accession_clean = accession.replace('-', '')
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession}.txt"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Failed to download {accession}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading {accession}: {e}")
            return None
    
    def extract_items(self, filing_text):
        """Extract individual 8-K items from filing text."""
        if not filing_text:
            return {}
        
        # Clean HTML and entities
        filing_text = re.sub(r'<[^>]+>', ' ', filing_text)
        filing_text = re.sub(r'&[a-z]+;', ' ', filing_text)
        filing_text = re.sub(r'&#?\d+;?', ' ', filing_text)
        filing_text = re.sub(r'&nbsp;?', ' ', filing_text)
        
        items = {}
        
        # Find all item headers (multiple patterns for robustness)
        patterns = [
            r'(?i)\n\s*item\s*(\d+\.?\d*)[:\s\-]*([^\n]*)',
            r'(?i)^item\s*(\d+\.?\d*)[:\s\-]*([^\n]*)',
            r'(?i)\bitem\s+(\d+\.?\d*)[:\s\-]*([^\n]*)'
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.finditer(pattern, filing_text, re.MULTILINE)
            all_matches.extend(matches)
        
        # Remove duplicates and sort by position
        seen_positions = set()
        unique_matches = []
        for match in all_matches:
            if match.start() not in seen_positions:
                seen_positions.add(match.start())
                unique_matches.append(match)
        
        unique_matches.sort(key=lambda x: x.start())
        
        # Extract text for each item
        for i, match in enumerate(unique_matches):
            item_num_raw = match.group(1)
            
            # Normalize item number (e.g., "2.2" -> "2.02", "2" -> "2.01")
            if '.' not in item_num_raw:
                item_num = item_num_raw + '.01'
            else:
                parts = item_num_raw.split('.')
                if len(parts[1]) == 1:
                    item_num = f"{parts[0]}.0{parts[1]}"
                else:
                    item_num = f"{parts[0]}.{parts[1][:2]}"
            
            # Skip invalid item numbers
            if not re.match(r'^\d{1,2}\.\d{2}$', item_num):
                continue
            
            # Extract text between this item and next
            start_pos = match.end()
            
            # Find end position
            if i < len(unique_matches) - 1:
                end_pos = unique_matches[i + 1].start()
            else:
                # Look for signature section
                sig_match = re.search(r'(?i)\n\s*signatures?\s*\n', filing_text[start_pos:])
                if sig_match:
                    end_pos = start_pos + sig_match.start()
                else:
                    end_pos = min(start_pos + 20000, len(filing_text))
            
            # Extract and clean text
            item_text = filing_text[start_pos:end_pos].strip()
            item_text = re.sub(r'\s+', ' ', item_text)
            
            # Only keep if substantial (>200 chars with >50 letters)
            if len(item_text) > 200 and len(re.findall(r'[a-zA-Z]', item_text)) > 50:
                # Limit to 10000 chars per item (more generous than before)
                items[item_num] = item_text[:10000]
        
        return items
    
    def process_filing(self, row, text_dict):
        """Process a single filing to extract items."""
        results = []
        
        # ALWAYS download full filing for proper item separation
        # The existing text data is pre-extracted single items, not full filings
        logger.debug(f"Downloading full filing for {row['accession']}")
        filing_text = self.download_filing(row['cik'], row['accession'])
        time.sleep(0.1)  # Rate limiting
        
        # Fall back to existing text if download fails
        if not filing_text:
            filing_text = text_dict.get(row['accession'], '')
        
        # Extract items
        items = self.extract_items(filing_text)
        
        if not items:
            logger.debug(f"No items extracted from {row['accession']}")
            # Still create a record with the mixed text if available
            if filing_text:
                results.append({
                    'ticker': row['ticker'],
                    'filing_date': row['filing_date'],
                    'accession': row['accession'],
                    'item_number': 'MIXED',
                    'item_description': 'Mixed/Unstructured Content',
                    'item_text': filing_text[:10000],
                    'adjusted_return_pct': row.get('adjusted_return_pct', np.nan),
                    'outperformed_market': row.get('outperformed_market', np.nan)
                })
        else:
            # Create a record for each item
            for item_num, item_text in items.items():
                results.append({
                    'ticker': row['ticker'],
                    'filing_date': row['filing_date'],
                    'accession': row['accession'],
                    'item_number': item_num,
                    'item_description': self.item_descriptions.get(item_num, 'Unknown'),
                    'item_text': item_text,
                    'adjusted_return_pct': row.get('adjusted_return_pct', np.nan),
                    'outperformed_market': row.get('outperformed_market', np.nan)
                })
        
        return results
    
    def process_year(self, year, batch_size=100):
        """Process all filings for a year."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {year}")
        logger.info(f"{'='*60}")
        
        # Load data
        df_returns, text_dict = self.load_year_data(year)
        if df_returns is None:
            return
        
        # Create output directory
        output_dir = self.base_dir / str(year)
        output_dir.mkdir(exist_ok=True)
        
        # Process all filings
        all_results = []
        total_filings = len(df_returns)
        
        logger.info(f"Processing {total_filings} filings...")
        
        # Process in batches with progress bar
        with tqdm(total=total_filings, desc=f"Year {year}") as pbar:
            for idx, row in df_returns.iterrows():
                filing_results = self.process_filing(row, text_dict)
                all_results.extend(filing_results)
                
                # Update progress
                pbar.update(1)
                
                # Save intermediate results every batch_size filings
                if (idx + 1) % batch_size == 0:
                    self.save_results(all_results, output_dir, year, interim=True)
                    logger.debug(f"Saved interim results: {len(all_results)} items from {idx+1} filings")
        
        # Save final results
        self.save_results(all_results, output_dir, year, interim=False)
        
        # Print statistics
        df_final = pd.DataFrame(all_results)
        self.print_statistics(df_final, year)
        
        return df_final
    
    def save_results(self, results, output_dir, year, interim=False):
        """Save results in items_separated.csv format."""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        # Sort by ticker and filing date
        df = df.sort_values(['ticker', 'filing_date', 'item_number'])
        
        # Save main CSV (without text for readability)
        suffix = '_interim' if interim else ''
        
        # Save full data
        output_file = output_dir / f"items_separated{suffix}.csv"
        df.to_csv(output_file, index=False)
        
        # Also save a version without text for quick analysis
        df_no_text = df.drop(columns=['item_text'])
        metadata_file = output_dir / f"items_metadata{suffix}.csv"
        df_no_text.to_csv(metadata_file, index=False)
        
        logger.info(f"Saved {len(df)} items to {output_file}")
    
    def print_statistics(self, df, year):
        """Print processing statistics."""
        logger.info(f"\n{'='*40}")
        logger.info(f"Statistics for {year}")
        logger.info(f"{'='*40}")
        
        # Basic stats
        n_filings = df['accession'].nunique()
        n_items = len(df)
        
        logger.info(f"Total filings processed: {n_filings}")
        logger.info(f"Total items extracted: {n_items}")
        logger.info(f"Average items per filing: {n_items/n_filings:.2f}")
        
        # Item distribution
        item_counts = df['item_number'].value_counts()
        logger.info("\nTop 10 most common items:")
        for item, count in item_counts.head(10).items():
            desc = self.item_descriptions.get(item, 'Unknown')
            pct = count / n_items * 100
            logger.info(f"  {item}: {count} ({pct:.1f}%) - {desc}")
        
        # Return statistics
        if 'adjusted_return_pct' in df.columns:
            df_valid = df.dropna(subset=['adjusted_return_pct'])
            if len(df_valid) > 0:
                logger.info(f"\nReturn statistics:")
                logger.info(f"  Mean adjusted return: {df_valid['adjusted_return_pct'].mean():.2f}%")
                logger.info(f"  Std adjusted return: {df_valid['adjusted_return_pct'].std():.2f}%")
                logger.info(f"  Outperformed market: {df_valid['outperformed_market'].mean()*100:.1f}%")
    
    def process_all_years(self):
        """Process all specified years."""
        logger.info("="*60)
        logger.info("8-K ITEM SEPARATION PROCESSING")
        logger.info("="*60)
        logger.info(f"Years to process: {self.years}")
        
        for year in self.years:
            try:
                self.process_year(year)
            except Exception as e:
                logger.error(f"Error processing {year}: {e}")
                continue
        
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved in: {self.base_dir}/")
        
        # Print summary across all years
        self.print_overall_summary()
    
    def print_overall_summary(self):
        """Print summary statistics across all years."""
        logger.info("\nOVERALL SUMMARY")
        logger.info("="*40)
        
        all_items = []
        for year in self.years:
            file_path = self.base_dir / str(year) / "items_separated.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['year'] = year
                all_items.append(df)
        
        if all_items:
            df_all = pd.concat(all_items, ignore_index=True)
            
            logger.info(f"Total items across all years: {len(df_all)}")
            logger.info(f"Total unique filings: {df_all['accession'].nunique()}")
            
            # Items by year
            logger.info("\nItems by year:")
            for year in self.years:
                year_data = df_all[df_all['year'] == year]
                if len(year_data) > 0:
                    logger.info(f"  {year}: {len(year_data)} items from {year_data['accession'].nunique()} filings")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process 8-K filings with item separation')
    parser.add_argument('--years', nargs='+', type=int, 
                       default=[2021, 2022, 2023, 2024],
                       help='Years to process (default: 2021-2024)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode - process only first 10 filings per year')
    args = parser.parse_args()
    
    # Initialize processor
    processor = EightKItemProcessor(years=args.years)
    
    if args.test:
        logger.info("TEST MODE: Processing only first 10 filings per year")
        # Modify to process only subset
        for year in args.years:
            df_returns, text_dict = processor.load_year_data(year)
            if df_returns is not None:
                # Process only first 10
                df_subset = df_returns.head(10)
                results = []
                for _, row in df_subset.iterrows():
                    results.extend(processor.process_filing(row, text_dict))
                
                # Save test results
                output_dir = processor.base_dir / str(year)
                output_dir.mkdir(exist_ok=True)
                processor.save_results(results, output_dir, year, interim=False)
                
                df_results = pd.DataFrame(results)
                processor.print_statistics(df_results, year)
    else:
        # Full processing
        processor.process_all_years()


if __name__ == "__main__":
    main()