"""
Test extraction with 20 samples using real OpenAI API
Run this to test the pipeline before full-scale extraction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from extract_llm_features_final import LLMFeatureExtractor
import logging
import os

# Suppress HTTP request logs from OpenAI
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
os.environ["HTTPX_LOG_LEVEL"] = "WARNING"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Testing with 20 Samples - Real API")
    logger.info("=" * 60)
    
    # Load data
    data_path = Path('../phase1_ml/data/train.csv')
    df = pd.read_csv(data_path)
    
    # Sample 20 random rows
    sample_df = df.sample(n=20, random_state=42)
    logger.info(f"Selected 20 random samples from {len(df)} total")
    
    # Initialize extractor with real API
    logger.info("\nInitializing with OpenAI API...")
    extractor = LLMFeatureExtractor(use_mock=False)
    
    # Process samples
    logger.info("\nProcessing 20 samples...")
    results = extractor.process_batch(sample_df)
    
    # Save results
    output_path = Path('test_20_api_results.csv')
    results.to_csv(output_path, index=False)
    
    # Report
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Processed: {len(results)} rows")
    logger.info(f"✅ Features added: {len(results.columns) - len(sample_df.columns)}")
    logger.info(f"✅ Saved to: {output_path}")
    logger.info(f"\nCost: ${extractor.total_cost:.4f}")
    logger.info(f"Tokens: {extractor.total_tokens:,}")
    
    # Estimate full cost
    full_size = 14449  # filtered_train size
    estimated_cost = (extractor.total_cost / 20) * full_size
    logger.info(f"\nEstimated cost for full train set: ${estimated_cost:.2f}")


if __name__ == "__main__":
    main()