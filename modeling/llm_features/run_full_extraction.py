"""
Run full-scale LLM feature extraction for all datasets
Processes train, validation, and test sets in parallel batches
"""

import pandas as pd
from pathlib import Path
from extract_llm_features_final import LLMFeatureExtractor
import logging
import json
from datetime import datetime
import os

# Suppress HTTP request logs from OpenAI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
os.environ["HTTPX_LOG_LEVEL"] = "WARNING"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_dataset(dataset_name: str, extractor: LLMFeatureExtractor):
    """Process a single dataset (train/val/test)"""
    
    # Load data
    input_path = Path(f'../phase1_ml/data/{dataset_name}.csv')
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {dataset_name.upper()} Dataset")
    logger.info(f"{'='*60}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    # Check for existing checkpoint
    checkpoint_path = Path(f'checkpoint_{dataset_name}.csv')
    start_index = 0
    all_results = []
    
    if checkpoint_path.exists():
        checkpoint_df = pd.read_csv(checkpoint_path)
        start_index = len(checkpoint_df)
        all_results.append(checkpoint_df)
        logger.info(f"⚡ Resuming from checkpoint: {start_index} rows already processed")
    else:
        logger.info("Starting fresh processing (no checkpoint found)")
    
    # Process in batches
    batch_size = 100  # Increased from 50 to 100
    checkpoint_frequency = 25  # Save every 25 batches (2500 rows)
    
    for i in range(start_index, len(df), batch_size):
        batch_num = i // batch_size + 1
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        logger.info(f"Batch {batch_num}/{total_batches} ({dataset_name})")
        
        batch = df.iloc[i:i+batch_size]
        batch_results = extractor.process_batch(batch)
        all_results.append(batch_results)
        
        # Save checkpoint every N batches or every 500 rows (whichever comes first)
        should_checkpoint = (
            (batch_num % checkpoint_frequency == 0) or  # Every 50 batches
            (i % 500 == 0 and i > 0) or  # Every 500 rows
            (batch_num == total_batches)  # Last batch
        )
        
        if should_checkpoint:
            checkpoint_df = pd.concat(all_results, ignore_index=True)
            checkpoint_path = Path(f'checkpoint_{dataset_name}.csv')
            checkpoint_df.to_csv(checkpoint_path, index=False)
            logger.info(f"✓ Checkpoint saved: {len(checkpoint_df)} rows processed")
            logger.info(f"  Cost so far: ${extractor.total_cost:.2f}")
            logger.info(f"  Tokens used: {extractor.total_tokens:,}")
    
    # Combine and save final results
    final_df = pd.concat(all_results, ignore_index=True)
    output_path = Path(f'enhanced_{dataset_name}.csv')
    final_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ {dataset_name.upper()} complete: {len(final_df)} rows")
    logger.info(f"✅ Saved to: {output_path}")
    
    return True


def main():
    """Process all datasets"""
    start_time = datetime.now()
    
    logger.info("="*60)
    logger.info("FULL-SCALE LLM FEATURE EXTRACTION")
    logger.info("="*60)
    logger.info(f"Start time: {start_time}")
    
    # Initialize extractor once for all datasets
    logger.info("\nInitializing LLM Feature Extractor...")
    extractor = LLMFeatureExtractor(use_mock=False)
    
    # Process each dataset
    datasets = ['train', 'val', 'test']
    success_count = 0
    
    for dataset in datasets:
        if process_dataset(dataset, extractor):
            success_count += 1
        else:
            logger.error(f"Failed to process {dataset}")
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Datasets processed: {success_count}/{len(datasets)}")
    logger.info(f"Total cost: ${extractor.total_cost:.2f}")
    logger.info(f"Total tokens: {extractor.total_tokens:,}")
    logger.info(f"Total time: {duration}")
    logger.info(f"Success rate: {extractor.success_count}/{extractor.success_count + extractor.error_count}")
    
    # Save metrics
    metrics = {
        'datasets_processed': success_count,
        'total_cost': extractor.total_cost,
        'total_tokens': extractor.total_tokens,
        'duration_seconds': duration.total_seconds(),
        'success_tasks': extractor.success_count,
        'error_tasks': extractor.error_count,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('extraction_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\nMetrics saved to extraction_metrics.json")
    
    # Expected file sizes
    logger.info("\nOutput files created:")
    for dataset in datasets:
        output_file = Path(f'enhanced_{dataset}.csv')
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"  - {output_file}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()