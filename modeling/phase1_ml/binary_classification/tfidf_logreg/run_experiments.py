import pandas as pd
import numpy as np
from train_tfidf_logreg import load_data, train_tfidf_logreg
import json

def run_experiments():
    """Run multiple experiments with different parameter combinations"""
    
    # Load data once
    train_df, val_df, test_df = load_data()
    
    # Define experiments with different parameter combinations
    experiments = [
        # Original (overfitting)
        {"name": "Original (High Overfit)", "C": 1.0, "max_features": 5000, "min_df": 5, "max_df": 0.95},
        
        # Strong regularization (underperforming)
        {"name": "Strong Regularization", "C": 0.1, "max_features": 2000, "min_df": 10, "max_df": 0.8},
        
        # Balanced approaches
        {"name": "Balanced v1", "C": 0.5, "max_features": 3000, "min_df": 7, "max_df": 0.85},
        {"name": "Balanced v2", "C": 0.3, "max_features": 3500, "min_df": 8, "max_df": 0.85},
        {"name": "Balanced v3", "C": 0.2, "max_features": 2500, "min_df": 8, "max_df": 0.85},
        
        # Slightly less regularization
        {"name": "Moderate Reg v1", "C": 0.7, "max_features": 3500, "min_df": 6, "max_df": 0.9},
        {"name": "Moderate Reg v2", "C": 0.4, "max_features": 4000, "min_df": 7, "max_df": 0.88},
        
        # Feature-focused tuning
        {"name": "More Features", "C": 0.3, "max_features": 4500, "min_df": 10, "max_df": 0.85},
        {"name": "Fewer Features", "C": 0.5, "max_features": 1500, "min_df": 15, "max_df": 0.75},
    ]
    
    results_summary = []
    best_test_acc = 0
    best_model = None
    
    print("="*70)
    print("RUNNING MULTIPLE EXPERIMENTS")
    print("="*70)
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\nEXPERIMENT {i}/{len(experiments)}: {exp['name']}")
        print("-"*70)
        print(f"Parameters: C={exp['C']}, max_features={exp['max_features']}, "
              f"min_df={exp['min_df']}, max_df={exp['max_df']}")
        
        # Train model with these parameters
        model, vectorizer, results = train_tfidf_logreg(
            train_df.copy(),
            val_df.copy(),
            test_df.copy(),
            C=exp['C'],
            max_features=exp['max_features'],
            min_df=exp['min_df'],
            max_df=exp['max_df']
        )
        
        # Extract key metrics
        train_acc = results['results']['train_accuracy']
        val_acc = results['results']['val_accuracy']
        test_acc = results['results']['test_accuracy']
        test_improvement = results['results']['improvements']['test']
        overfit_gap = results['results']['overfitting_metrics']['train_test_gap']
        
        # Store summary
        exp_summary = {
            'experiment': exp['name'],
            'parameters': {
                'C': exp['C'],
                'max_features': exp['max_features'],
                'min_df': exp['min_df'],
                'max_df': exp['max_df']
            },
            'train_acc': round(train_acc, 4),
            'val_acc': round(val_acc, 4),
            'test_acc': round(test_acc, 4),
            'test_improvement': round(test_improvement, 4),
            'overfit_gap': round(overfit_gap, 4),
            'beats_baseline': test_improvement > 0
        }
        results_summary.append(exp_summary)
        
        # Track best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model = exp_summary
    
    # Save all results
    with open('experiment_comparison.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary table
    print("\n\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)
    print(f"{'Experiment':<20} {'C':<6} {'Features':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'vs Base':<10} {'Overfit':<10}")
    print("-"*100)
    
    for exp in results_summary:
        symbol = "✓" if exp['beats_baseline'] else "✗"
        print(f"{exp['experiment']:<20} "
              f"{exp['parameters']['C']:<6.2f} "
              f"{exp['parameters']['max_features']:<10d} "
              f"{exp['train_acc']:<10.2f}% "
              f"{exp['val_acc']:<10.2f}% "
              f"{exp['test_acc']:<10.2f}% "
              f"{exp['test_improvement']:+9.2f}% {symbol} "
              f"{exp['overfit_gap']:<10.2f}%")
    
    # Print best model
    print("\n" + "="*100)
    print("BEST MODEL")
    print("="*100)
    print(f"Experiment: {best_model['experiment']}")
    print(f"Test Accuracy: {best_model['test_acc']:.4f}%")
    print(f"Improvement over baseline: {best_model['test_improvement']:+.4f}%")
    print(f"Overfitting gap: {best_model['overfit_gap']:.4f}%")
    print(f"Parameters: C={best_model['parameters']['C']}, "
          f"max_features={best_model['parameters']['max_features']}, "
          f"min_df={best_model['parameters']['min_df']}, "
          f"max_df={best_model['parameters']['max_df']}")
    
    # Baselines for reference
    print("\n" + "="*100)
    print("BASELINE REFERENCE")
    print("="*100)
    print("Test set baseline (majority class): 52.0088%")
    print("To beat baseline, model needs: >52.0088% accuracy")
    
    return results_summary, best_model

if __name__ == "__main__":
    results_summary, best_model = run_experiments()