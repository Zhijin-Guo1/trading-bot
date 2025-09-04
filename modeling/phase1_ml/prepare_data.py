import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_gpt_summaries(years=[2014, 2015, 2016, 2017, 2018, 2019]):
    """Load all GPT summaries from the results folder"""
    all_summaries = []
    
    for year in years:
        summary_path = f'../../summarization/gpt_summarization/results/{year}/all_summaries.json'
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summaries = json.load(f)
                all_summaries.extend(summaries)
                print(f"Loaded {len(summaries)} summaries from {year}")
    
    df_summaries = pd.DataFrame(all_summaries)
    print(f"\nTotal summaries loaded: {len(df_summaries)}")
    return df_summaries

def load_enhanced_datasets(years=[2014, 2015, 2016, 2017, 2018, 2019]):
    """Load all enhanced datasets"""
    all_datasets = []
    
    for year in years:
        dataset_path = f'../../data/data_processing/enhanced_eight_k/enhanced_dataset_{year}_{year}_with_prices.csv'
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            all_datasets.append(df)
            print(f"Loaded {len(df)} records from enhanced dataset {year}")
    
    df_enhanced = pd.concat(all_datasets, ignore_index=True)
    print(f"\nTotal enhanced records loaded: {len(df_enhanced)}")
    return df_enhanced

def merge_datasets(df_summaries, df_enhanced):
    """Merge GPT summaries with enhanced dataset based on accession number"""
    
    # Merge on accession number
    df_merged = pd.merge(
        df_summaries,
        df_enhanced,
        on='accession',
        how='inner',
        suffixes=('_summary', '_enhanced')
    )
    
    print(f"\nMerged dataset size: {len(df_merged)}")
    
    # Select relevant columns
    columns_to_keep = [
        'accession',
        'ticker_enhanced',  # Use ticker from enhanced dataset
        'filing_date_enhanced',  # Use filing date from enhanced dataset
        'summary',  # GPT summary
        'items_present',  # 8-K items
        'item_count',
        'text_length_enhanced',  # Original text length
        'sector',
        'industry',
        'momentum_7d',
        'momentum_30d',
        'momentum_90d',
        'momentum_365d',
        'vix_level',
        'filing_time',
        'adjusted_return_pct',
        'immediate_reaction_pct'
    ]
    
    # Check which columns exist
    available_columns = [col for col in columns_to_keep if col in df_merged.columns]
    df_final = df_merged[available_columns].copy()
    
    # Rename columns for clarity
    df_final.rename(columns={
        'ticker_enhanced': 'ticker',
        'filing_date_enhanced': 'filing_date',
        'text_length_enhanced': 'original_text_length'
    }, inplace=True)
    
    return df_final

def create_classification_targets(df):
    """Create binary and three-class classification targets"""
    
    # Binary classification (using 0% threshold)
    df['binary_target'] = (df['adjusted_return_pct'] > 0).astype(int)
    
    # Three-class classification (UP >1%, STAY ±1%, DOWN <-1%)
    conditions = [
        df['adjusted_return_pct'] > 1.0,
        df['adjusted_return_pct'] < -1.0
    ]
    choices = [2, 0]  # 2=UP, 0=DOWN
    df['three_class_target'] = np.select(conditions, choices, default=1)  # 1=STAY
    
    # Map to readable labels
    df['binary_label'] = df['binary_target'].map({0: 'DOWN', 1: 'UP'})
    df['three_class_label'] = df['three_class_target'].map({0: 'DOWN', 1: 'STAY', 2: 'UP'})
    
    return df

def create_temporal_splits(df, val_ratio=0.15, test_ratio=0.15):
    """Create temporal train/val/test splits"""
    
    # Convert filing_date to datetime
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    
    # Sort by date
    df = df.sort_values('filing_date').reset_index(drop=True)
    
    # Calculate split points
    n = len(df)
    train_end = int(n * (1 - val_ratio - test_ratio))
    val_end = int(n * (1 - test_ratio))
    
    # Create splits
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Get date ranges
    print("\nTemporal Split Information:")
    print(f"Train: {train_df['filing_date'].min().date()} to {train_df['filing_date'].max().date()} ({len(train_df)} samples)")
    print(f"Val:   {val_df['filing_date'].min().date()} to {val_df['filing_date'].max().date()} ({len(val_df)} samples)")
    print(f"Test:  {test_df['filing_date'].min().date()} to {test_df['filing_date'].max().date()} ({len(test_df)} samples)")
    
    return train_df, val_df, test_df

def analyze_class_distribution(df, label_col, split_name):
    """Analyze class distribution in a dataset"""
    print(f"\n{split_name} Class Distribution ({label_col}):")
    distribution = df[label_col].value_counts(normalize=True).sort_index()
    for label, pct in distribution.items():
        count = df[label_col].value_counts()[label]
        print(f"  {label}: {count} samples ({pct:.2%})")
    return distribution

def save_datasets(train_df, val_df, test_df, output_dir='data'):
    """Save processed datasets"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    # Save as JSON for easier text processing
    train_df.to_json(f'{output_dir}/train.json', orient='records', indent=2)
    val_df.to_json(f'{output_dir}/val.json', orient='records', indent=2)
    test_df.to_json(f'{output_dir}/test.json', orient='records', indent=2)
    
    print(f"\nDatasets saved to {output_dir}/")
    print(f"  - train.csv/json: {len(train_df)} samples")
    print(f"  - val.csv/json: {len(val_df)} samples")
    print(f"  - test.csv/json: {len(test_df)} samples")

def create_dataset_statistics(train_df, val_df, test_df):
    """Create comprehensive statistics about the dataset"""
    
    stats = {
        'dataset_sizes': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
            'total': len(train_df) + len(val_df) + len(test_df)
        },
        'date_ranges': {
            'train': {
                'start': str(train_df['filing_date'].min().date()),
                'end': str(train_df['filing_date'].max().date())
            },
            'val': {
                'start': str(val_df['filing_date'].min().date()),
                'end': str(val_df['filing_date'].max().date())
            },
            'test': {
                'start': str(test_df['filing_date'].min().date()),
                'end': str(test_df['filing_date'].max().date())
            }
        },
        'binary_distribution': {},
        'three_class_distribution': {},
        'return_statistics': {}
    }
    
    # Binary class distribution
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        dist = df['binary_label'].value_counts(normalize=True).to_dict()
        stats['binary_distribution'][split_name] = {k: float(v) for k, v in dist.items()}
    
    # Three-class distribution
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        dist = df['three_class_label'].value_counts(normalize=True).to_dict()
        stats['three_class_distribution'][split_name] = {k: float(v) for k, v in dist.items()}
    
    # Return statistics
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        stats['return_statistics'][split_name] = {
            'mean': float(df['adjusted_return_pct'].mean()),
            'std': float(df['adjusted_return_pct'].std()),
            'min': float(df['adjusted_return_pct'].min()),
            'max': float(df['adjusted_return_pct'].max()),
            'median': float(df['adjusted_return_pct'].median())
        }
    
    # Save statistics
    with open('data/dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nDataset statistics saved to data/dataset_statistics.json")
    
    return stats

def main():
    print("="*50)
    print("Data Preparation for Stock Movement Prediction")
    print("="*50)
    
    # Load data
    print("\n1. Loading GPT summaries...")
    df_summaries = load_gpt_summaries()
    
    print("\n2. Loading enhanced datasets...")
    df_enhanced = load_enhanced_datasets()
    
    # Merge datasets
    print("\n3. Merging datasets...")
    df_merged = merge_datasets(df_summaries, df_enhanced)
    
    # Create classification targets
    print("\n4. Creating classification targets...")
    df_merged = create_classification_targets(df_merged)
    
    # Create temporal splits
    print("\n5. Creating temporal train/val/test splits...")
    train_df, val_df, test_df = create_temporal_splits(df_merged)
    
    # Analyze class distributions
    print("\n6. Analyzing class distributions...")
    
    # Binary classification
    analyze_class_distribution(train_df, 'binary_label', 'Train')
    analyze_class_distribution(val_df, 'binary_label', 'Validation')
    analyze_class_distribution(test_df, 'binary_label', 'Test')
    
    # Three-class classification
    analyze_class_distribution(train_df, 'three_class_label', 'Train')
    analyze_class_distribution(val_df, 'three_class_label', 'Validation')
    analyze_class_distribution(test_df, 'three_class_label', 'Test')
    
    # Save datasets
    print("\n7. Saving processed datasets...")
    save_datasets(train_df, val_df, test_df)
    
    # Create statistics
    print("\n8. Creating dataset statistics...")
    stats = create_dataset_statistics(train_df, val_df, test_df)
    
    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)
    
    # Print recommendations
    print("\nRecommendations based on class distribution:")
    
    # Check binary balance
    train_binary_dist = train_df['binary_label'].value_counts(normalize=True)
    if abs(train_binary_dist.get('UP', 0) - train_binary_dist.get('DOWN', 0)) < 0.1:
        print("✓ Binary classification is relatively balanced")
    else:
        print("⚠ Binary classification shows imbalance - consider using class weights")
    
    # Check three-class balance
    train_three_dist = train_df['three_class_label'].value_counts(normalize=True)
    if train_three_dist.get('STAY', 0) > 0.5:
        print("⚠ Three-class has high STAY proportion - binary classification may be more suitable")
    else:
        print("✓ Three-class distribution is reasonable")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = main()