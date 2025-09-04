#!/usr/bin/env python3
"""
Test different regularization strategies for Random Forest
"""

from train_combined_features import CombinedFeatureClassifier, test_rf_regularization
import pandas as pd

def main():
    print("="*60)
    print("RANDOM FOREST REGULARIZATION TESTING")
    print("="*60)
    
    # Load data
    classifier = CombinedFeatureClassifier()
    train_df, val_df, test_df = classifier.load_data(use_filtered=True)
    
    # Test different regularization levels
    results = test_rf_regularization(train_df, val_df, test_df)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    baseline = 52.0
    print(f"\nBaseline accuracy: {baseline:.2f}%")
    print("\nKey Findings:")
    
    for level, metrics in results.items():
        improvement = metrics['test_acc'] - baseline
        if improvement > 0:
            print(f"  ✓ {level}: {metrics['test_acc']:.2f}% (beats baseline by {improvement:.2f}%)")
        else:
            print(f"  ✗ {level}: {metrics['test_acc']:.2f}% (below baseline by {abs(improvement):.2f}%)")
        print(f"    - Overfitting gap: {metrics['overfit_gap']:.2f}%")

if __name__ == "__main__":
    main()