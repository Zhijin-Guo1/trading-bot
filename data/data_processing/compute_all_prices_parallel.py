#!/usr/bin/env python3
"""
Compute prices for all years (2014-2019) in parallel.
Includes 5-day returns, immediate reactions, all normalized by SPY.
"""

import multiprocessing as mp
from pathlib import Path
import pandas as pd
from datetime import datetime
import time
from compute_price import PriceComputer
import sys


def process_single_year(year: int) -> dict:
    """
    Process a single year's data.
    Returns status dictionary.
    """
    try:
        print(f"[{year}] Starting processing...")
        start_time = time.time()
        
        # Create a new PriceComputer instance for this process
        pc = PriceComputer()
        
        # Process the year
        result_df = pc.process_year(year)
        
        elapsed = time.time() - start_time
        
        if result_df is not None:
            # Check output file
            output_file = Path(f"/Users/engs2742/trading-bot/data_processing/enhanced_eight_k/enhanced_dataset_{year}_{year}_with_prices.csv")
            
            return {
                'year': year,
                'status': 'success',
                'records': len(result_df),
                'elapsed_time': elapsed,
                'output_file': str(output_file),
                'file_exists': output_file.exists(),
                'statistics': pc.stats
            }
        else:
            return {
                'year': year,
                'status': 'failed',
                'records': 0,
                'elapsed_time': elapsed,
                'error': 'Failed to process year'
            }
            
    except Exception as e:
        return {
            'year': year,
            'status': 'error',
            'records': 0,
            'elapsed_time': 0,
            'error': str(e)
        }


def run_parallel_processing(years: list = None):
    """
    Run price computation for multiple years in parallel.
    """
    if years is None:
        years = [2014, 2015, 2016, 2017, 2018, 2019]
    
    print("="*70)
    print("PARALLEL PRICE COMPUTATION")
    print("="*70)
    print(f"Years to process: {years}")
    print(f"Number of CPU cores: {mp.cpu_count()}")
    print(f"Processes to use: {min(len(years), mp.cpu_count())}")
    print("="*70)
    
    # Record start time
    overall_start = time.time()
    
    # Create a pool of workers
    num_processes = min(len(years), mp.cpu_count())
    
    print(f"\nStarting {num_processes} parallel processes...\n")
    
    # Use Pool for parallel processing
    with mp.Pool(processes=num_processes) as pool:
        # Map the process function to each year
        results = pool.map(process_single_year, years)
    
    # Calculate total time
    total_elapsed = time.time() - overall_start
    
    # Display results
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    
    # Summary statistics
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] != 'success')
    total_records = sum(r.get('records', 0) for r in results)
    
    print(f"\n📊 Summary:")
    print(f"  Total years processed: {len(years)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total records: {total_records:,}")
    print(f"  Total time: {total_elapsed:.1f} seconds")
    print(f"  Average time per year: {total_elapsed/len(years):.1f} seconds")
    
    # Detailed results per year
    print(f"\n📁 Results by Year:")
    print("-"*60)
    
    for result in sorted(results, key=lambda x: x['year']):
        year = result['year']
        status = result['status']
        
        if status == 'success':
            emoji = "✅"
            records = result.get('records', 0)
            elapsed = result.get('elapsed_time', 0)
            stats = result.get('statistics', {})
            
            print(f"{emoji} {year}: {records:,} records in {elapsed:.1f}s")
            
            # Show filing time distribution if available
            if stats:
                total = stats.get('total_processed', 0)
                if total > 0:
                    pre = stats.get('pre_market', 0)
                    market = stats.get('market_hours', 0)
                    after = stats.get('after_hours', 0)
                    overnight = stats.get('overnight', 0)
                    
                    print(f"     Filing times: Pre-market:{pre}, Market:{market}, After:{after}, Overnight:{overnight}")
            
            # Check if file exists
            if result.get('file_exists'):
                print(f"     Output: {Path(result['output_file']).name}")
            else:
                print(f"     ⚠️  Output file not found!")
                
        else:
            emoji = "❌"
            error = result.get('error', 'Unknown error')
            print(f"{emoji} {year}: Failed - {error}")
    
    print("\n" + "="*70)
    
    # List all output files
    output_dir = Path("/Users/engs2742/trading-bot/data_processing/enhanced_eight_k")
    output_files = sorted(output_dir.glob("*_with_prices.csv"))
    
    if output_files:
        print("\n📂 Output Files Created:")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    return results


def main():
    """Main entry point."""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            years = [2014, 2015, 2016, 2017, 2018, 2019]
        else:
            try:
                years = [int(y) for y in sys.argv[1:]]
            except ValueError:
                print("Error: Invalid year format")
                print("Usage: python compute_all_prices_parallel.py [all | year1 year2 ...]")
                return
    else:
        print("Usage:")
        print("  python compute_all_prices_parallel.py all         # Process all years")
        print("  python compute_all_prices_parallel.py 2018 2019   # Process specific years")
        
        response = input("\nProcess all years 2014-2019? (yes/no): ")
        if response.lower() == 'yes':
            years = [2014, 2015, 2016, 2017, 2018, 2019]
        else:
            print("Cancelled.")
            return
    
    # Run parallel processing
    run_parallel_processing(years)


if __name__ == "__main__":
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()