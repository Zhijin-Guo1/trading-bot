"""
Direct 8-K Filing Summarization using gpt-5-mini with parallel processing - 2016 Data
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
import logging
from openai import OpenAI
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'rag_system'))

from rag_system.config.settings import (
    OPENAI_API_KEY, SOURCE_DATA_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Rate limiter to avoid overwhelming the API
class RateLimiter:
    def __init__(self, max_concurrent=10):
        self.semaphore = threading.Semaphore(max_concurrent)
    
    def __enter__(self):
        self.semaphore.acquire()
        return self
    
    def __exit__(self, *args):
        self.semaphore.release()

rate_limiter = RateLimiter(max_concurrent=10)  # Allow 10 concurrent API calls


class ParallelDirectSummarizer:
    """Direct summarization with parallel processing"""

    def __init__(self, num_workers=10):
        self.num_workers = num_workers
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(__file__).parent / "results" / f"parallel_direct_{self.run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = "gpt-5-mini"
        self.total_cost = 0
        self.total_tokens = 0
        self.results = []

        logger.info(f"Parallel Direct Summarizer initialized")
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output: {self.output_dir}")

    def load_all_2016_data(self, n_samples: Optional[int] = None) -> List[Dict]:
        """Load filings and labels"""
        print("\n" + "=" * 60)
        print("LOADING 2016 DATA FOR PARALLEL SUMMARIZATION")
        print("=" * 60)

        texts_file = SOURCE_DATA_DIR / "texts" / "merged_texts_2016.json"
        labels_file = SOURCE_DATA_DIR / "enhanced_dataset_2016_2016.csv"

        print(f"Loading from: {texts_file}")
        with open(texts_file, 'r') as f:
            all_texts = json.load(f)

        print(f"Loading labels from: {labels_file}")
        labels_df = pd.read_csv(labels_file)

        filings = []
        for i, (accession, data) in enumerate(all_texts.items()):
            if n_samples and i >= n_samples:
                break

            label_row = labels_df[labels_df['accession'] == accession]
            if not label_row.empty:
                row = label_row.iloc[0]
                data['accession'] = accession
                data['signal'] = row['signal']
                data['adjusted_return_pct'] = row['adjusted_return_pct']
                filings.append(data)

        print(f"Loaded {len(filings)} filings with labels")

        # Save filing metadata
        filing_info = pd.DataFrame([
            {
                'accession': f['accession'],
                'ticker': f['ticker'],
                'date': f['filing_date'],
                'signal': f['signal'],
                'return': f['adjusted_return_pct']
            }
            for f in filings
        ])
        filing_info.to_csv(self.output_dir / "filing_metadata.csv", index=False)

        return filings

    def summarize_filing(self, filing: Dict) -> Dict:
        """Summarize a single filing with rate limiting"""
        
        input_text = f"""You are a financial analyst. Analyze this 8-K filing and create a comprehensive summary.

Company: {filing['ticker']}
Filing Date: {filing['filing_date']}

8-K Filing:
{filing['text']}

Create a concise summary (200-300 words) that captures:
1. The most important events or announcements
2. Any financial metrics or changes
3. Material impacts on the business
4. Forward-looking statements or guidance

Focus on information that would impact stock price. Be specific about numbers and facts.

Summary:"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with rate_limiter:  # Rate limiting
                    response = client.responses.create(
                        model="gpt-5-mini",
                        input=input_text,
                        reasoning={
                            "effort": "minimal"
                        }
                    )

                # Extract response content
                if hasattr(response, 'output'):
                    if isinstance(response.output, list) and len(response.output) > 0:
                        for item in response.output:
                            if hasattr(item, 'content') and item.content:
                                if isinstance(item.content, list) and len(item.content) > 0:
                                    summary = item.content[0].text if hasattr(item.content[0], 'text') else str(item.content[0])
                                else:
                                    summary = str(item.content)
                                break
                        else:
                            summary = str(response.output)
                    else:
                        summary = str(response.output)
                else:
                    summary = str(response)
                
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else len(input_text) // 4
                cost = tokens * 0.25 / 1000000

                return {
                    'accession': filing['accession'],
                    'ticker': filing['ticker'],
                    'filing_date': filing['filing_date'],
                    'true_label': filing['signal'],
                    'true_return': filing['adjusted_return_pct'],
                    'summary': str(summary),
                    'method': 'parallel_gpt-5-mini',
                    'tokens_used': int(tokens),
                    'cost': float(cost),
                    'text_length': len(filing['text']),
                    'status': 'success'
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed {filing['ticker']}: {str(e)[:50]}")
                    return {
                        'accession': filing['accession'],
                        'ticker': filing['ticker'],
                        'filing_date': filing['filing_date'],
                        'true_label': filing['signal'],
                        'true_return': filing['adjusted_return_pct'],
                        'summary': f'Error: {str(e)[:100]}',
                        'status': 'error'
                    }

    def process_batch_parallel(self, batch_filings: List[Dict]) -> List[Dict]:
        """Process a batch of filings in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_filing = {
                executor.submit(self.summarize_filing, filing): filing 
                for filing in batch_filings
            }
            
            # Process as completed with progress bar
            with tqdm(total=len(batch_filings), desc="Processing batch") as pbar:
                for future in as_completed(future_to_filing):
                    filing = future_to_filing[future]
                    try:
                        result = future.result(timeout=60)
                        results.append(result)
                        
                        # Update totals
                        if result['status'] == 'success':
                            self.total_tokens += result['tokens_used']
                            self.total_cost += result['cost']
                    except Exception as e:
                        logger.error(f"Error processing {filing['ticker']}: {e}")
                        results.append({
                            'accession': filing['accession'],
                            'ticker': filing['ticker'],
                            'filing_date': filing['filing_date'],
                            'true_label': filing['signal'],
                            'true_return': filing['adjusted_return_pct'],
                            'summary': f'Processing error: {str(e)[:100]}',
                            'status': 'error'
                        })
                    pbar.update(1)

        return results

    def run_full_pipeline(self, n_samples: Optional[int] = None):
        """Run the complete parallel summarization pipeline"""
        start_time = time.time()
        
        print("\n" + "=" * 60)
        print(f"PARALLEL DIRECT SUMMARIZATION PIPELINE (gpt-5-mini)")
        print(f"Workers: {self.num_workers}")
        print("=" * 60)

        # Load data
        filings = self.load_all_2016_data(n_samples)

        # Process in batches
        batch_size = 50
        all_results = []

        print("\n" + "=" * 60)
        print("PROCESSING FILINGS")
        print("=" * 60)

        for i in range(0, len(filings), batch_size):
            batch = filings[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(filings) + batch_size - 1) // batch_size

            print(f"\n{'=' * 40}")
            print(f"Batch {batch_num}/{total_batches} ({len(batch)} filings)")
            print(f"{'=' * 40}")

            batch_start = time.time()
            
            # Process batch in parallel
            batch_results = self.process_batch_parallel(batch)
            all_results.extend(batch_results)
            
            batch_time = time.time() - batch_start

            # Save checkpoint
            checkpoint_file = self.output_dir / f"checkpoint_batch_{batch_num}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            # Stats
            successful = sum(1 for r in batch_results if r['status'] == 'success')
            failed = sum(1 for r in batch_results if r['status'] == 'error')

            print(f"Batch complete: {successful} success, {failed} failed")
            print(f"Batch time: {batch_time:.1f}s ({batch_time/len(batch):.1f}s per filing)")
            print(f"Total processed so far: {len(all_results)}")

        # Calculate total time
        total_time = time.time() - start_time
        
        # Save final results
        self.save_final_results(all_results, total_time)

        return all_results

    def save_final_results(self, results: List[Dict], total_time: float):
        """Save final results with timing statistics"""
        print("\n" + "=" * 60)
        print("SAVING FINAL RESULTS")
        print("=" * 60)

        # Full results
        with open(self.output_dir / "all_summaries.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Simplified version
        simplified = [
            {
                'accession': r['accession'],
                'ticker': r['ticker'],
                'date': r['filing_date'],
                'summary': r.get('summary', 'No summary'),
                'true_label': r['true_label'],
                'true_return': r['true_return']
            }
            for r in results if r['status'] == 'success'
        ]

        with open(self.output_dir / "summaries_simplified.json", 'w') as f:
            json.dump(simplified, f, indent=2)

        # CSV format
        df = pd.DataFrame(simplified)
        df.to_csv(self.output_dir / "summaries.csv", index=False)

        # Statistics
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')

        labels = [r['true_label'] for r in results if r['status'] == 'success']
        returns = [r['true_return'] for r in results if r['status'] == 'success']

        stats = {
            'method': 'parallel_gpt-5-mini',
            'num_workers': self.num_workers,
            'total_processed': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) * 100 if results else 0,
            'label_distribution': {
                'UP': labels.count('UP'),
                'DOWN': labels.count('DOWN'),
                'STAY': labels.count('STAY')
            },
            'return_stats': {
                'min': min(returns) if returns else 0,
                'max': max(returns) if returns else 0,
                'avg': sum(returns) / len(returns) if returns else 0,
                'std': np.std(returns) if returns else 0
            },
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_cost_per_filing': self.total_cost / len(results) if results else 0,
            'timing': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'avg_time_per_filing': total_time / len(results) if results else 0,
                'filings_per_minute': len(results) / (total_time / 60) if total_time > 0 else 0
            }
        }

        with open(self.output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nResults saved to: {self.output_dir}")
        print(f"\nFinal Statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
        print(f"  Failed: {stats['failed']}")
        print(f"  Total cost: ${stats['total_cost']:.2f}")
        print(f"  Avg cost per filing: ${stats['avg_cost_per_filing']:.4f}")
        
        print(f"\nTiming Statistics:")
        print(f"  Total time: {stats['timing']['total_time_minutes']:.1f} minutes")
        print(f"  Avg time per filing: {stats['timing']['avg_time_per_filing']:.1f} seconds")
        print(f"  Processing rate: {stats['timing']['filings_per_minute']:.1f} filings/minute")
        
        print(f"\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count}")


def main():
    """Run the parallel direct summarization pipeline"""
    # Initialize with 10 parallel workers
    summarizer = ParallelDirectSummarizer(num_workers=10)

    # Process all 2016 data
    # Use n_samples=None to process all available data
    results = summarizer.run_full_pipeline(n_samples=None)

    print("\n" + "=" * 60)
    print("PARALLEL DIRECT SUMMARIZATION COMPLETE!")
    print(f"Processed {len(results)} filings")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()