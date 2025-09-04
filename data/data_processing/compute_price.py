#!/usr/bin/env python3
"""
Compute stock prices and returns based on exact filing times.
Handles different filing time scenarios appropriately.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta, time
import pytz
import numpy as np
from tqdm import tqdm
import requests
from typing import Dict, Optional, Tuple
import time as time_module
import warnings
warnings.filterwarnings('ignore')


class PriceComputer:
    def __init__(self):
        self.base_dir = Path("/Users/engs2742/trading-bot/data_processing")
        self.enhanced_dir = self.base_dir / "enhanced_eight_k"
        
        # Market hours (Eastern Time)
        self.market_tz = pytz.timezone('US/Eastern')
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.market_close = time(16, 0)  # 4:00 PM ET
        self.after_hours_close = time(20, 0)  # 8:00 PM ET
        
        # Cache for filing times and price data
        self.filing_times_cache = {}
        self.intraday_cache = {}
        self.daily_cache = {}
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'pre_market': 0,
            'market_hours': 0,
            'after_hours': 0,
            'overnight': 0
        }
    
    def get_filing_time(self, accession: str, cik: str = None) -> Optional[datetime]:
        """
        Fetch exact filing time from SEC EDGAR.
        
        Args:
            accession: SEC accession number (e.g., "0001193125-14-406296")
            cik: Company CIK (optional)
        
        Returns:
            datetime object with filing time in ET, or None if not found
        """
        if accession in self.filing_times_cache:
            return self.filing_times_cache[accession]
        
        try:
            # Extract CIK from accession if not provided
            # Accession format: XXXXXXXXXX-YY-ZZZZZZ where X is padded CIK
            if not cik:
                # Keep the padded CIK from accession for URL construction
                cik_padded = accession.split('-')[0]
                cik = cik_padded.lstrip('0')
            else:
                cik_padded = cik.zfill(10)
            
            # Format accession without dashes for URL
            acc_no_dash = accession.replace('-', '')
            
            # Direct URL to filing header - use unpadded CIK for folder, padded accession for file
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dash}/{accession}.txt"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Your Company Name admin@yourcompany.com)'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                content = response.text
                
                # Look for ACCEPTANCE-DATETIME in header (SEC format)
                for line in content.split('\n')[:50]:
                    if '<ACCEPTANCE-DATETIME>' in line:
                        # Extract timestamp between tags
                        start = line.find('<ACCEPTANCE-DATETIME>') + len('<ACCEPTANCE-DATETIME>')
                        end = line.find('</ACCEPTANCE-DATETIME>', start) if '</ACCEPTANCE-DATETIME>' in line else len(line)
                        accepted_str = line[start:end].strip()
                        
                        # Format: YYYYMMDDHHMMSS
                        if len(accepted_str) >= 14:
                            filing_datetime = datetime(
                                int(accepted_str[0:4]),   # year
                                int(accepted_str[4:6]),   # month
                                int(accepted_str[6:8]),   # day
                                int(accepted_str[8:10]),  # hour
                                int(accepted_str[10:12]), # minute
                                int(accepted_str[12:14])  # second
                            )
                            
                            # Cache the result
                            self.filing_times_cache[accession] = filing_datetime
                            
                            # Rate limit to respect SEC servers (10 requests per second max)
                            time_module.sleep(0.1)
                            
                            return filing_datetime
            
            return None
            
        except Exception as e:
            print(f"Error fetching filing time for {accession}: {e}")
            return None
    
    def classify_filing_time(self, filing_datetime: datetime) -> str:
        """
        Classify filing time into categories.
        
        Returns:
            'pre_market', 'market_hours', 'after_hours', or 'overnight'
        """
        if not filing_datetime:
            return 'after_hours'  # Default conservative assumption
        
        # Convert to ET if needed
        if filing_datetime.tzinfo is None:
            filing_datetime = self.market_tz.localize(filing_datetime)
        else:
            filing_datetime = filing_datetime.astimezone(self.market_tz)
        
        filing_time = filing_datetime.time()
        
        # Check if weekend
        if filing_datetime.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return 'overnight'
        
        # Classify by time
        if time(4, 0) <= filing_time < self.market_open:
            return 'pre_market'
        elif self.market_open <= filing_time < self.market_close:
            return 'market_hours'
        elif self.market_close <= filing_time < self.after_hours_close:
            return 'after_hours'
        else:
            return 'overnight'
    
    def get_intraday_entry_price(self, ticker: str, filing_datetime: datetime, 
                                 interval: str = '5m') -> Optional[Tuple[float, datetime]]:
        """
        Get the next available intraday price after filing time.
        Used for filings during market hours.
        
        Args:
            ticker: Stock ticker
            filing_datetime: Exact filing time
            interval: Time interval for intraday data ('1m', '5m', '15m')
        
        Returns:
            Tuple of (price, timestamp) or None
        """
        try:
            # Download intraday data for the filing day
            stock = yf.Ticker(ticker)
            
            # Get data for filing day and next day (in case filing is near close)
            start_date = filing_datetime.date()
            end_date = start_date + timedelta(days=2)
            
            # Download intraday data
            intraday_data = stock.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if intraday_data.empty:
                return None
            
            # Find first price after filing time
            filing_aware = filing_datetime
            if filing_aware.tzinfo is None:
                filing_aware = self.market_tz.localize(filing_aware)
            
            # Filter for prices after filing
            after_filing = intraday_data[intraday_data.index > filing_aware]
            
            if not after_filing.empty:
                first_bar = after_filing.iloc[0]
                # Use the closing price of the interval (more reliable than open)
                return first_bar['Close'], after_filing.index[0]
            
            return None
            
        except Exception as e:
            print(f"Error getting intraday data for {ticker}: {e}")
            return None
    
    def calculate_immediate_reaction(self, ticker_data: pd.DataFrame, spy_data: pd.DataFrame,
                                    filing_date: datetime, filing_time_category: str,
                                    entry_date: datetime) -> Optional[Dict]:
        """
        Calculate immediate market reaction after filing release.
        
        Formulas:
        - After-market filing: (Next Open - Today Close) / Today Close
        - During-market filing: (Today Close - Pre-filing Price) / Pre-filing Price  
        - Pre-market filing: (Today Open - Yesterday Close) / Yesterday Close
        
        All normalized by SPY movement for the same period.
        """
        try:
            reaction_data = {}
            
            if filing_time_category in ['after_hours', 'overnight']:
                # After-market: Next Open - Filing Day Close
                # Find filing day (or last trading day before filing)
                filing_day = None
                for td in reversed(spy_data.index):
                    if td.date() <= filing_date.date():
                        filing_day = td
                        break
                
                if filing_day and entry_date:
                    # Stock reaction
                    stock_close_before = ticker_data.loc[filing_day, 'Close']
                    stock_open_after = ticker_data.loc[entry_date, 'Open']
                    stock_reaction = (stock_open_after - stock_close_before) / stock_close_before * 100
                    
                    # SPY reaction
                    spy_close_before = spy_data.loc[filing_day, 'Close']
                    spy_open_after = spy_data.loc[entry_date, 'Open']
                    spy_reaction = (spy_open_after - spy_close_before) / spy_close_before * 100
                    
                    reaction_data['immediate_reaction_pct'] = round(stock_reaction - spy_reaction, 4)
                    reaction_data['stock_overnight_gap_pct'] = round(stock_reaction, 4)
                    reaction_data['spy_overnight_gap_pct'] = round(spy_reaction, 4)
                    reaction_data['reaction_type'] = 'overnight_gap'
            
            elif filing_time_category == 'market_hours':
                # During market: Today Close - Last price before filing
                # This would need intraday data for precision
                # For now, use Open as proxy for pre-filing price
                filing_day = None
                for td in spy_data.index:
                    if td.date() == filing_date.date():
                        filing_day = td
                        break
                
                if filing_day:
                    # Approximate: use Open as pre-filing price
                    stock_open = ticker_data.loc[filing_day, 'Open']
                    stock_close = ticker_data.loc[filing_day, 'Close']
                    stock_reaction = (stock_close - stock_open) / stock_open * 100
                    
                    spy_open = spy_data.loc[filing_day, 'Open']
                    spy_close = spy_data.loc[filing_day, 'Close']
                    spy_reaction = (spy_close - spy_open) / spy_open * 100
                    
                    reaction_data['immediate_reaction_pct'] = round(stock_reaction - spy_reaction, 4)
                    reaction_data['stock_intraday_move_pct'] = round(stock_reaction, 4)
                    reaction_data['spy_intraday_move_pct'] = round(spy_reaction, 4)
                    reaction_data['reaction_type'] = 'intraday_move'
            
            elif filing_time_category == 'pre_market':
                # Pre-market: Today Open - Yesterday Close
                filing_day = None
                for td in spy_data.index:
                    if td.date() == filing_date.date():
                        filing_day = td
                        break
                
                if filing_day:
                    # Find previous trading day
                    prev_day = None
                    for i, td in enumerate(spy_data.index):
                        if td == filing_day and i > 0:
                            prev_day = spy_data.index[i-1]
                            break
                    
                    if prev_day:
                        stock_prev_close = ticker_data.loc[prev_day, 'Close']
                        stock_today_open = ticker_data.loc[filing_day, 'Open']
                        stock_reaction = (stock_today_open - stock_prev_close) / stock_prev_close * 100
                        
                        spy_prev_close = spy_data.loc[prev_day, 'Close']
                        spy_today_open = spy_data.loc[filing_day, 'Open']
                        spy_reaction = (spy_today_open - spy_prev_close) / spy_prev_close * 100
                        
                        reaction_data['immediate_reaction_pct'] = round(stock_reaction - spy_reaction, 4)
                        reaction_data['stock_open_gap_pct'] = round(stock_reaction, 4)
                        reaction_data['spy_open_gap_pct'] = round(spy_reaction, 4)
                        reaction_data['reaction_type'] = 'open_gap'
            
            return reaction_data if reaction_data else None
            
        except Exception as e:
            print(f"Error calculating immediate reaction: {e}")
            return None
    
    def get_entry_exit_prices(self, ticker: str, filing_date: str, accession: str,
                            spy_data: pd.DataFrame, stock_data: Dict) -> Optional[Dict]:
        """
        Calculate entry and exit prices based on filing time, including immediate reaction.
        
        Returns:
            Dictionary with prices and metadata, or None if calculation fails
        """
        try:
            filing_date = pd.to_datetime(filing_date)
            
            # Check if we have stock data
            if ticker not in stock_data or stock_data[ticker].empty:
                return None
            
            ticker_data = stock_data[ticker]
            trading_days = spy_data.index
            
            # Get exact filing time from SEC
            filing_datetime = self.get_filing_time(accession)
            
            if filing_datetime:
                # Use exact filing time to classify
                filing_time_category = self.classify_filing_time(filing_datetime)
            else:
                # Fallback: assume after-hours (conservative)
                filing_time_category = 'after_hours'
            
            # Determine entry date and price based on filing time
            if filing_time_category == 'pre_market':
                # Same day open
                entry_date = None
                for td in trading_days:
                    if td.date() >= filing_date.date():
                        entry_date = td
                        break
                if entry_date and entry_date.date() == filing_date.date():
                    entry_price_type = 'Open'
                else:
                    # Filing was on non-trading day, use next day open
                    entry_price_type = 'Open'
            
            elif filing_time_category == 'market_hours':
                # Would use intraday price in production
                # For now, use next day open (conservative)
                entry_date = None
                for td in trading_days:
                    if td.date() > filing_date.date():
                        entry_date = td
                        break
                entry_price_type = 'Open'
            
            else:  # after_hours or overnight
                # Next trading day open
                entry_date = None
                for td in trading_days:
                    if td.date() > filing_date.date():
                        entry_date = td
                        break
                entry_price_type = 'Open'
            
            if not entry_date:
                return None
            
            # Find exit date (5 trading days from entry)
            try:
                entry_idx = trading_days.get_loc(entry_date)
                if entry_idx + 4 >= len(trading_days):
                    return None
                exit_date = trading_days[entry_idx + 4]
            except:
                return None
            
            # Verify dates exist in both datasets
            if entry_date not in ticker_data.index or entry_date not in spy_data.index:
                return None
            if exit_date not in ticker_data.index or exit_date not in spy_data.index:
                return None
            
            # Get prices
            stock_entry = ticker_data.loc[entry_date, entry_price_type]
            spy_entry = spy_data.loc[entry_date, entry_price_type]
            
            # Exit always at close
            stock_exit = ticker_data.loc[exit_date, 'Close']
            spy_exit = spy_data.loc[exit_date, 'Close']
            
            # Validate prices
            if any(pd.isna([stock_entry, stock_exit, spy_entry, spy_exit])):
                return None
            if any(p <= 0 for p in [stock_entry, stock_exit, spy_entry, spy_exit]):
                return None
            
            # Calculate returns
            stock_return = (stock_exit - stock_entry) / stock_entry * 100
            spy_return = (spy_exit - spy_entry) / spy_entry * 100
            adjusted_return = stock_return - spy_return
            
            # Calculate immediate reaction
            reaction_data = self.calculate_immediate_reaction(
                ticker_data, spy_data, filing_date, 
                filing_time_category, entry_date
            )
            
            # Update statistics
            self.stats[filing_time_category] += 1
            
            # Build result dictionary
            result = {
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'stock_entry_price': round(stock_entry, 2),
                'stock_exit_price': round(stock_exit, 2),
                'spy_entry_price': round(spy_entry, 2),
                'spy_exit_price': round(spy_exit, 2),
                'stock_return_pct': round(stock_return, 4),
                'spy_return_pct': round(spy_return, 4),
                'adjusted_return_pct': round(adjusted_return, 4),
                'filing_time_category': filing_time_category,
                'entry_price_type': entry_price_type
            }
            
            # Add immediate reaction data if available
            if reaction_data:
                result.update(reaction_data)
            
            return result
            
        except Exception as e:
            print(f"Error calculating prices for {ticker} on {filing_date}: {e}")
            return None
    
    def process_year(self, year: int) -> pd.DataFrame:
        """
        Process all filings for a given year.
        """
        print(f"\n{'='*60}")
        print(f"Processing Year {year}")
        print(f"{'='*60}")
        
        # Load existing data
        csv_file = self.enhanced_dir / f"enhanced_dataset_{year}_{year}.csv"
        if not csv_file.exists():
            print(f"File not found: {csv_file}")
            return None
        
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records from {csv_file.name}")
        
        # Download market data
        print(f"\nDownloading market data for {year}...")
        spy = yf.Ticker("SPY")
        spy_data = spy.history(start=f"{year-1}-12-01", end=f"{year+1}-02-01")
        
        if spy_data.empty:
            print(f"ERROR: No SPY data for {year}")
            return None
        
        spy_data.index = spy_data.index.tz_localize(None)
        
        # Download stock data
        unique_tickers = df['ticker'].unique()
        print(f"Downloading data for {len(unique_tickers)} tickers...")
        
        stock_data = {}
        for ticker in tqdm(unique_tickers, desc="Downloading"):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=f"{year-1}-12-01", end=f"{year+1}-02-01")
                
                if not data.empty:
                    data.index = data.index.tz_localize(None)
                    stock_data[ticker] = data
            except:
                continue
        
        print(f"Successfully downloaded: {len(stock_data)}/{len(unique_tickers)} tickers")
        
        # Add new columns if they don't exist
        new_columns = [
            'entry_date', 'exit_date',
            'stock_entry_price', 'stock_exit_price',
            'spy_entry_price', 'spy_exit_price',
            'stock_return_pct', 'spy_return_pct',
            'adjusted_return_pct',
            'filing_time_category', 'entry_price_type',
            # Immediate reaction columns
            'immediate_reaction_pct',
            'stock_overnight_gap_pct', 'spy_overnight_gap_pct',
            'stock_intraday_move_pct', 'spy_intraday_move_pct',
            'stock_open_gap_pct', 'spy_open_gap_pct',
            'reaction_type'
        ]
        
        for col in new_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Process each filing
        print(f"\nCalculating prices for {len(df)} filings...")
        successful = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            ticker = row['ticker']
            filing_date = row['filing_date']
            accession = row['accession']
            
            # Calculate prices with exact filing time
            result = self.get_entry_exit_prices(ticker, filing_date, accession, spy_data, stock_data)
            
            if result:
                # Update dataframe
                for key, value in result.items():
                    df.at[idx, key] = value
                successful += 1
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
            
            self.stats['total_processed'] += 1
        
        # Save updated dataframe
        output_file = self.enhanced_dir / f"enhanced_dataset_{year}_{year}_with_prices.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ Processed {successful}/{len(df)} filings successfully")
        print(f"📁 Saved to: {output_file.name}")
        
        # Show filing time distribution
        print(f"\nFiling Time Distribution:")
        print(f"  Pre-market:    {self.stats['pre_market']:,}")
        print(f"  Market hours:  {self.stats['market_hours']:,}")
        print(f"  After-hours:   {self.stats['after_hours']:,}")
        print(f"  Overnight:     {self.stats['overnight']:,}")
        
        return df
    
    def run(self, years: list = None):
        """
        Run price computation for specified years.
        """
        if years is None:
            years = [2019]  # Default to 2019 for testing
        
        print(f"\n{'='*60}")
        print(f"STOCK PRICE COMPUTATION")
        print(f"{'='*60}")
        print(f"Years to process: {years}")
        print(f"\nPrice Calculation Logic:")
        print(f"  • Pre-market filings → Same day OPEN")
        print(f"  • Market hours filings → Next available price (conservative: next OPEN)")
        print(f"  • After-hours filings → Next day OPEN")
        print(f"  • Exit → Always 5th trading day CLOSE")
        
        results = []
        for year in years:
            df = self.process_year(year)
            if df is not None:
                results.append(df)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total processed: {self.stats['total_processed']:,}")
        print(f"Successful: {self.stats['successful']:,}")
        print(f"Failed: {self.stats['failed']:,}")
        
        if self.stats['total_processed'] > 0:
            success_rate = self.stats['successful'] / self.stats['total_processed'] * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        return results


def main():
    """Main execution function."""
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            years = list(range(2014, 2020))  # 2014-2019
        else:
            years = [int(y) for y in sys.argv[1:]]
    else:
        print("Usage:")
        print("  python compute_price.py 2019              # Single year")
        print("  python compute_price.py 2018 2019         # Multiple years")
        print("  python compute_price.py all               # All years (2014-2019)")
        return
    
    # Initialize and run
    computer = PriceComputer()
    computer.run(years=years)


if __name__ == "__main__":
    main()