import pandas as pd
import numpy as np
import json

def calculate_baseline_performance():
    """Calculate baseline performance for both binary and three-class classification"""
    
    # Load dataset statistics
    with open('next_day_data/dataset_statistics.json', 'r') as f:
        stats = json.load(f)
    
    # Load actual data for exact counts
    train_df = pd.read_csv('next_day_data/train.csv')
    val_df = pd.read_csv('next_day_data/val.csv')
    test_df = pd.read_csv('next_day_data/test.csv')
    
    print("="*60)
    print("BASELINE MODEL ANALYSIS")
    print("="*60)
    
    # Binary Classification Baseline
    print("\n1. BINARY CLASSIFICATION BASELINE (Majority Class Predictor)")
    print("-"*60)
    
    baseline_results = {}
    
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # Calculate exact counts
        total = len(df)
        up_count = (df['binary_label'] == 'UP').sum()
        down_count = (df['binary_label'] == 'DOWN').sum()
        
        # Determine majority class and its accuracy
        if up_count >= down_count:
            majority_class = 'UP'
            majority_count = up_count
        else:
            majority_class = 'DOWN'
            majority_count = down_count
        
        majority_accuracy = majority_count / total * 100
        
        # Store for summary
        baseline_results[split_name] = {
            'majority_class': majority_class,
            'accuracy': majority_accuracy
        }
        
        print(f"\n{split_name.upper()} Set:")
        print(f"  Total samples: {total:,}")
        print(f"  UP samples: {up_count:,} ({up_count/total*100:.4f}%)")
        print(f"  DOWN samples: {down_count:,} ({down_count/total*100:.4f}%)")
        print(f"  Majority class: {majority_class}")
        print(f"  Baseline Accuracy: {majority_accuracy:.4f}%")
        print(f"  Random Guess (50/50): 50.0000%")
    
    # Three-Class Classification Baseline
    print("\n\n2. THREE-CLASS CLASSIFICATION BASELINE (Majority Class Predictor)")
    print("-"*60)
    
    three_class_results = {}
    
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # Calculate exact counts
        total = len(df)
        up_count = (df['three_class_label'] == 'UP').sum()
        down_count = (df['three_class_label'] == 'DOWN').sum()
        stay_count = (df['three_class_label'] == 'STAY').sum()
        
        # Determine majority class
        class_counts = {'UP': up_count, 'DOWN': down_count, 'STAY': stay_count}
        majority_class = max(class_counts, key=class_counts.get)
        majority_count = class_counts[majority_class]
        majority_accuracy = majority_count / total * 100
        
        # Store for summary
        three_class_results[split_name] = {
            'majority_class': majority_class,
            'accuracy': majority_accuracy
        }
        
        print(f"\n{split_name.upper()} Set:")
        print(f"  Total samples: {total:,}")
        print(f"  UP samples: {up_count:,} ({up_count/total*100:.4f}%)")
        print(f"  DOWN samples: {down_count:,} ({down_count/total*100:.4f}%)")
        print(f"  STAY samples: {stay_count:,} ({stay_count/total*100:.4f}%)")
        print(f"  Majority class: {majority_class}")
        print(f"  Baseline Accuracy: {majority_accuracy:.4f}%")
        print(f"  Random Guess (uniform): 33.3333%")
    
    # Alternative Baselines
    print("\n\n3. ALTERNATIVE BASELINE STRATEGIES")
    print("-"*60)
    
    # Always predict "market follows" (no outperformance)
    print("\nAlways Predict 'Follow Market' (return ≈ 0%):")
    train_stay_pct = stats['three_class_distribution']['train'].get('STAY', 0)
    print(f"  Would achieve ~{train_stay_pct:.2%} accuracy in three-class")
    
    # Historical mean return
    print("\nHistorical Mean Return Strategy:")
    for split in ['train', 'val', 'test']:
        mean_return = stats['return_statistics'][split]['mean']
        print(f"  {split}: mean return = {mean_return:.2%}")
        if mean_return > 0:
            print(f"    → Always predicting UP would be rational")
        else:
            print(f"    → Always predicting DOWN would be rational")
    
    # SPY benchmark (buy and hold)
    print("\n\n4. IMPORTANT CONSIDERATIONS FOR MODEL EVALUATION")
    print("-"*60)
    print("""
    Binary Classification:
    - Majority class baseline: ~50-52% (nearly random)
    - Any model must beat 52% to add value
    - Consider precision/recall trade-offs for trading strategy
    
    Three-Class Classification:
    - Majority class baseline: ~34-38% (better than random 33.33%)
    - STAY class (±1%) is hardest to predict
    - Consider collapsing to binary if STAY performance is poor
    
    Trading Performance Metrics (beyond accuracy):
    - Sharpe Ratio (risk-adjusted returns)
    - Maximum Drawdown
    - Win Rate vs Profit Factor
    - Transaction costs impact
    """)
    
    # Add summary of exact baselines
    print("\n\n5. EXACT BASELINE SUMMARY FOR BINARY CLASSIFICATION")
    print("-"*60)
    print("\nTo beat the baseline, your model must achieve:")
    for split_name, result in baseline_results.items():
        print(f"  {split_name.upper():5} set: > {result['accuracy']:.4f}% (majority class: {result['majority_class']})")
    
    print("\n\n6. EXACT BASELINE SUMMARY FOR THREE-CLASS CLASSIFICATION")
    print("-"*60)
    print("\nTo beat the baseline, your model must achieve:")
    for split_name, result in three_class_results.items():
        print(f"  {split_name.upper():5} set: > {result['accuracy']:.4f}% (majority class: {result['majority_class']})")
    
    # Load actual data to calculate more baselines
    print("\n\n7. ACTUAL RETURN DISTRIBUTION ANALYSIS")
    print("-"*60)
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        returns = df['adjusted_return_pct']
        print(f"\n{name} Set Return Distribution:")
        print(f"  Positive returns: {(returns > 0).mean():.2%}")
        print(f"  Negative returns: {(returns < 0).mean():.2%}")
        print(f"  Within ±1%: {((returns >= -1) & (returns <= 1)).mean():.2%}")
        print(f"  Median return: {returns.median():.3f}%")
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        print(f"  Percentiles: ", end="")
        for p in percentiles:
            val = np.percentile(returns, p)
            print(f"{p}th={val:.2f}% ", end="")
        print()
    
    return stats

if __name__ == "__main__":
    stats = calculate_baseline_performance()