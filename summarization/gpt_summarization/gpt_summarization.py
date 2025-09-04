"""
Direct 8-K Filing Summarization using gpt-5-mini-2025-08-07
Processes full filings without RAG for comparison
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import settings from rag_system config
from rag_system.config.settings import (
    OPENAI_API_KEY, SOURCE_DATA_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class DirectFilingSummarizer:
    """Direct summarization of 8-K filings using gpt-5-mini-2025-08-07"""

    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use gpt_summarization folder for outputs
        self.output_dir = Path(__file__).parent / "results" / f"direct_summaries_{self.run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = "gpt-5-mini"
        self.total_cost = 0
        self.total_tokens = 0
        self.results = []

        logger.info(f"Direct Summarizer initialized")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output: {self.output_dir}")

    def load_all_2019_data(self, n_samples: Optional[int] = None) -> List[Dict]:
        """Load filings and labels using the same format as RAG pipeline"""
        print("\n" + "=" * 60)
        print("LOADING 2019 DATA FOR DIRECT SUMMARIZATION")
        print("=" * 60)

        texts_file = SOURCE_DATA_DIR / "texts" / "merged_texts_2019.json"
        labels_file = SOURCE_DATA_DIR / "enhanced_dataset_2019_2019.csv"

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
        """Summarize a single filing without truncation"""

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
                response = client.responses.create(
                    model="gpt-5-mini",
                    input=input_text,
                    reasoning={
                        "effort": "minimal"
                    }
                )

                # Extract response content and usage from GPT-5 response format
                # GPT-5 response has a complex structure with reasoning and output messages
                if hasattr(response, 'output'):
                    # Extract the actual text from the response
                    if isinstance(response.output, list) and len(response.output) > 0:
                        # Find the output message (not reasoning)
                        for item in response.output:
                            if hasattr(item, 'content') and item.content:
                                # Extract text from content
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
                
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else len(input_text) // 4  # Estimate if not provided

                # gpt-5 pricing (adjust as needed)
                cost = tokens * 0.25 / 1000000

                self.total_tokens += tokens
                self.total_cost += cost

                return {
                    'accession': filing['accession'],
                    'ticker': filing['ticker'],
                    'filing_date': filing['filing_date'],
                    'true_label': filing['signal'],
                    'true_return': filing['adjusted_return_pct'],
                    'summary': str(summary),  # Ensure it's a string
                    'method': 'direct_gpt-5-mini',
                    'tokens_used': int(tokens),  # Ensure it's an int
                    'cost': float(cost),  # Ensure it's a float
                    'text_length': len(filing['text']),
                    'status': 'success'
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retry {attempt + 1} for {filing['ticker']}: {str(e)[:50]}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to summarize {filing['accession']}: {e}")
                    return {
                        'accession': filing['accession'],
                        'ticker': filing['ticker'],
                        'filing_date': filing['filing_date'],
                        'true_label': filing['signal'],
                        'true_return': filing['adjusted_return_pct'],
                        'summary': f'Error: {str(e)[:100]}',
                        'status': 'error'
                    }

    def process_batch(self, batch_filings: List[Dict]) -> List[Dict]:
        """Process a batch of filings sequentially"""
        results = []

        for filing in tqdm(batch_filings, desc="Processing filings"):
            result = self.summarize_filing(filing)
            results.append(result)
            time.sleep(0.5)  # Rate limiting

        return results

    def run_full_pipeline(self, n_samples: Optional[int] = None):
        """Run the complete direct summarization pipeline"""
        print("\n" + "=" * 60)
        print("DIRECT SUMMARIZATION PIPELINE (gpt-5-mini)")
        print("=" * 60)

        # Load data
        filings = self.load_all_2019_data(n_samples)

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

            # Process batch
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)

            # Save checkpoint
            checkpoint_file = self.output_dir / f"checkpoint_batch_{batch_num}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            # Stats
            successful = sum(1 for r in batch_results if r['status'] == 'success')
            failed = sum(1 for r in batch_results if r['status'] == 'error')

            print(f"Batch complete: {successful} success, {failed} failed")

            # Delay between batches
            if i + batch_size < len(filings):
                time.sleep(2)

        # Save final results
        self.save_final_results(all_results)

        return all_results

    def save_final_results(self, results: List[Dict]):
        """Save final results in multiple formats (same as RAG pipeline)"""
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
            'method': 'direct_gpt-5-mini',
            'total_processed': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) * 100,
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
            'avg_cost_per_filing': self.total_cost / len(results) if results else 0
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
        print(f"\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count}")


def main():
    """Run the direct summarization pipeline"""
    summarizer = DirectFilingSummarizer()

    # Process all filings or a sample
    # For testing, use n_samples=100
    # For full dataset, use n_samples=None
    results = summarizer.run_full_pipeline(n_samples=100)

    print("\n" + "=" * 60)
    print("DIRECT SUMMARIZATION COMPLETE!")
    print(f"Processed {len(results)} filings")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()