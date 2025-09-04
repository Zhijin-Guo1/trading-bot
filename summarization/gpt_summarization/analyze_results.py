"""
Analyze GPT-5-mini summarization results across all years
"""
import json
from pathlib import Path
import pandas as pd

results_dir = Path('/Users/engs2742/trading-bot/gpt_summarization/results')
years = [2014, 2015, 2016, 2017, 2018, 2019]

print('='*80)
print('GPT-5-MINI SUMMARIZATION RESULTS - COMPREHENSIVE SUMMARY')
print('='*80)
print()

all_data = []
year_stats = []

for year in years:
    file_path = results_dir / str(year) / 'all_summaries.json'
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Filter successful entries
        successful = [d for d in data if d.get('status') == 'success']
        failed = [d for d in data if d.get('status') != 'success']
        
        # Calculate statistics
        labels = [d['true_label'] for d in successful]
        returns = [d['true_return'] for d in successful]
        costs = [d.get('cost', 0) for d in successful]
        tokens = [d.get('tokens_used', 0) for d in successful]
        
        stats = {
            'Year': year,
            'Total Filings': len(data),
            'Successful': len(successful),
            'Failed': len(failed),
            'Success Rate': f'{len(successful)/len(data)*100:.1f}%' if data else '0%',
            'UP': labels.count('UP'),
            'DOWN': labels.count('DOWN'), 
            'STAY': labels.count('STAY'),
            'Avg Return': f'{sum(returns)/len(returns):.2f}%' if returns else '0%',
            'Total Cost': f'${sum(costs):.2f}',
            'Avg Tokens': int(sum(tokens)/len(tokens)) if tokens else 0
        }
        
        year_stats.append(stats)
        all_data.extend(successful)

# Print year-by-year statistics
print('YEAR-BY-YEAR STATISTICS:')
print('-'*80)
for stats in year_stats:
    print(f"\nYear {stats['Year']}:")
    print(f"  • Total Filings: {stats['Total Filings']:,} ({stats['Successful']:,} successful, {stats['Failed']} failed)")
    print(f"  • Success Rate: {stats['Success Rate']}")
    print(f"  • Label Distribution: UP={stats['UP']:,}, DOWN={stats['DOWN']:,}, STAY={stats['STAY']:,}")
    print(f"  • Average Return: {stats['Avg Return']}")
    print(f"  • Total Cost: {stats['Total Cost']}")
    print(f"  • Avg Tokens/Filing: {stats['Avg Tokens']:,}")

# Overall statistics
print()
print('='*80)
print('OVERALL STATISTICS (2014-2019):')
print('-'*80)
total_filings = sum(s['Total Filings'] for s in year_stats)
total_successful = sum(s['Successful'] for s in year_stats)
total_failed = sum(s['Failed'] for s in year_stats)
total_cost = sum(float(s['Total Cost'].replace('$','')) for s in year_stats)

print(f'• Total Filings Processed: {total_filings:,}')
print(f'• Total Successful: {total_successful:,}')
print(f'• Total Failed: {total_failed:,}')
print(f'• Overall Success Rate: {total_successful/total_filings*100:.1f}%')
print(f'• Total Processing Cost: ${total_cost:.2f}')
print(f'• Average Cost per Filing: ${total_cost/total_filings:.4f}')

# Label distribution across all years
all_labels = [d['true_label'] for d in all_data]
print(f"\n• Overall Label Distribution:")
print(f"  - UP: {all_labels.count('UP'):,} ({all_labels.count('UP')/len(all_labels)*100:.1f}%)")
print(f"  - DOWN: {all_labels.count('DOWN'):,} ({all_labels.count('DOWN')/len(all_labels)*100:.1f}%)")
print(f"  - STAY: {all_labels.count('STAY'):,} ({all_labels.count('STAY')/len(all_labels)*100:.1f}%)")

# Data structure summary
print()
print('='*80)
print('DATA STRUCTURE IN all_summaries.json:')
print('-'*80)
print("Each JSON file contains an array of filing summaries with the following fields:")
print("• accession: SEC filing accession number (unique identifier)")
print("• ticker: Company stock ticker symbol")
print("• filing_date: Date of the 8-K filing")
print("• true_label: Actual stock movement (UP/DOWN/STAY)")
print("• true_return: Actual 5-day return percentage")
print("• summary: GPT-5-mini generated summary (200-300 words)")
print("• method: Processing method (parallel_gpt-5-mini)")
print("• tokens_used: Number of tokens consumed")
print("• cost: Processing cost in USD")
print("• text_length: Original filing text length in characters")
print("• status: Processing status (success/error)")

# Sample summary
print()
print('='*80)
print('SAMPLE SUMMARY (from 2019 data):')
print('-'*80)
if all_data:
    sample = all_data[0]
    print(f"Ticker: {sample['ticker']}")
    print(f"Date: {sample['filing_date']}")
    print(f"True Label: {sample['true_label']} ({sample['true_return']:.2f}%)")
    print(f"\nSummary (first 500 chars):")
    print(sample['summary'][:500] + "...")