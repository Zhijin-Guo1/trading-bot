import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import time
from datetime import datetime

def load_data():
    """Load train, validation, and test datasets"""
    train_df = pd.read_csv('../../filtered_data/filtered_train.csv')
    val_df = pd.read_csv('../../filtered_data/filtered_val.csv')
    test_df = pd.read_csv('../../filtered_data/filtered_test.csv')
    
    print(f"Data loaded:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    return train_df, val_df, test_df

def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text

def train_tfidf_logreg(train_df, val_df, test_df, C=0.1, max_features=2000, min_df=10, max_df=0.8):
    """Train TF-IDF + Logistic Regression model with regularization
    
    Args:
        C: Inverse regularization strength (smaller = more regularization)
        max_features: Maximum number of TF-IDF features
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
    """
    
    print("\n" + "="*60)
    print("TF-IDF + LOGISTIC REGRESSION MODEL (WITH REGULARIZATION)")
    print("="*60)
    
    # Preprocess text
    print("\n1. Preprocessing text...")
    train_df['processed_summary'] = train_df['summary'].apply(preprocess_text)
    val_df['processed_summary'] = val_df['summary'].apply(preprocess_text)
    test_df['processed_summary'] = test_df['summary'].apply(preprocess_text)
    
    # Extract features and labels
    X_train = train_df['processed_summary'].values
    y_train = train_df['binary_target'].values
    
    X_val = val_df['processed_summary'].values
    y_val = val_df['binary_target'].values
    
    X_test = test_df['processed_summary'].values
    y_test = test_df['binary_target'].values
    
    # Create TF-IDF vectorizer with more conservative parameters
    print("\n2. Creating TF-IDF features (with reduced dimensionality)...")
    print("   Hyperparameters:")
    print(f"   - max_features: {max_features} (reduced from 5000)")
    print(f"   - ngram_range: (1, 2)")
    print(f"   - min_df: {min_df} (increased from 5)")
    print(f"   - max_df: {max_df} (reduced from 0.95)")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,  # Reduced to prevent overfitting
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=min_df,  # Increased to filter out rare terms
        max_df=max_df,  # Reduced to filter out very common terms
        stop_words='english',
        sublinear_tf=True,  # Apply sublinear scaling to term frequency
        norm='l2'  # L2 normalization
    )
    
    # Fit on training data and transform
    start_time = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   TF-IDF shape: {X_train_tfidf.shape}")
    print(f"   Vectorization time: {time.time() - start_time:.2f}s")
    
    # Train Logistic Regression with stronger regularization
    print("\n3. Training Logistic Regression with L2 regularization...")
    print("   Hyperparameters:")
    print(f"   - C: {C} (reduced for stronger regularization)")
    print(f"   - penalty: l2")
    print(f"   - max_iter: 1000")
    print(f"   - solver: liblinear")
    print(f"   - class_weight: balanced (to handle slight class imbalance)")
    
    start_time = time.time()
    model = LogisticRegression(
        C=C,  # Reduced for stronger regularization
        penalty='l2',  # L2 regularization
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    
    model.fit(X_train_tfidf, y_train)
    print(f"   Training time: {time.time() - start_time:.2f}s")
    
    # Make predictions
    print("\n4. Making predictions...")
    train_pred = model.predict(X_train_tfidf)
    val_pred = model.predict(X_val_tfidf)
    test_pred = model.predict(X_test_tfidf)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100
    
    # Print results
    print("\n5. RESULTS")
    print("-"*60)
    
    # Baselines for comparison
    baselines = {
        'train': 50,
        'val': 52.7,
        'test': 52
    }
    
    print("\nAccuracy Comparison:")
    print(f"{'Split':<10} {'Baseline':<12} {'Model':<12} {'Improvement':<12}")
    print("-"*46)
    
    for split_name, acc, baseline in [
        ('Train', train_acc, baselines['train']),
        ('Val', val_acc, baselines['val']),
        ('Test', test_acc, baselines['test'])
    ]:
        improvement = acc - baseline
        symbol = "✓" if improvement > 0 else "✗"
        print(f"{split_name:<10} {baseline:>8.4f}%    {acc:>8.4f}%    {improvement:>+8.4f}% {symbol}")
    
    # Detailed classification report for test set
    print("\n\nTest Set Classification Report:")
    print("-"*60)
    print(classification_report(y_test, test_pred, target_names=['DOWN', 'UP']))
    
    # Confusion matrix
    print("Test Set Confusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(f"          Predicted")
    print(f"          DOWN    UP")
    print(f"Actual DOWN  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       UP    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Get feature importance (top positive and negative coefficients)
    print("\n6. TOP PREDICTIVE FEATURES")
    print("-"*60)
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Sort features by coefficient
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)
    
    print("\nTop 10 features for UP prediction:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['coefficient']:+.4f}")
    
    print("\nTop 10 features for DOWN prediction:")
    for idx, row in feature_importance.tail(10).iterrows():
        print(f"  {row['feature']:30s} {row['coefficient']:+.4f}")
    
    # Save model and vectorizer
    print("\n7. Saving model and vectorizer...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("   Model saved to: model.pkl")
    print("   Vectorizer saved to: vectorizer.pkl")
    
    # Calculate overfitting metrics
    overfitting_gap = train_acc - test_acc
    val_test_gap = abs(val_acc - test_acc)
    
    print("\n\n8. OVERFITTING ANALYSIS")
    print("-"*60)
    print(f"Train-Test Gap: {overfitting_gap:.4f}%")
    print(f"Val-Test Gap: {val_test_gap:.4f}%")
    if overfitting_gap > 10:
        print("⚠️  Warning: Significant overfitting detected (>10% gap)")
    elif overfitting_gap > 5:
        print("⚠️  Moderate overfitting detected (5-10% gap)")
    else:
        print("✓ Good generalization (<5% gap)")
    
    # Save results
    results = {
        'model': 'TF-IDF + Logistic Regression (Regularized)',
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': {
            'tfidf': {
                'max_features': max_features,
                'ngram_range': [1, 2],
                'min_df': min_df,
                'max_df': max_df,
                'sublinear_tf': True,
                'norm': 'l2'
            },
            'logreg': {
                'C': C,
                'penalty': 'l2',
                'max_iter': 1000,
                'solver': 'liblinear',
                'class_weight': 'balanced'
            }
        },
        'results': {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'improvements': {
                'train': train_acc - baselines['train'],
                'val': val_acc - baselines['val'],
                'test': test_acc - baselines['test']
            },
            'overfitting_metrics': {
                'train_test_gap': overfitting_gap,
                'val_test_gap': val_test_gap
            }
        },
        'feature_shape': X_train_tfidf.shape[1]
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("   Results saved to: results.json")
    
    return model, vectorizer, results

def hyperparameter_tuning(train_df, val_df, test_df):
    """Tune hyperparameters using validation set"""
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Define hyperparameter grid
    C_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    max_features_values = [1000, 2000, 3000]
    min_df_values = [5, 10, 20]
    
    best_val_acc = 0
    best_params = {}
    all_results = []
    
    print("\nTesting different hyperparameter combinations...")
    print("-"*60)
    
    for C in C_values:
        for max_features in max_features_values:
            for min_df in min_df_values:
                print(f"\nTesting: C={C}, max_features={max_features}, min_df={min_df}")
                
                # Train with current hyperparameters
                model, vectorizer, results = train_tfidf_logreg(
                    train_df.copy(), 
                    val_df.copy(), 
                    test_df.copy(),
                    C=C,
                    max_features=max_features,
                    min_df=min_df,
                    max_df=0.8
                )
                
                val_acc = results['results']['val_accuracy']
                train_test_gap = results['results']['overfitting_metrics']['train_test_gap']
                
                # Store results
                all_results.append({
                    'C': C,
                    'max_features': max_features,
                    'min_df': min_df,
                    'val_acc': val_acc,
                    'test_acc': results['results']['test_accuracy'],
                    'train_test_gap': train_test_gap
                })
                
                # Update best if validation accuracy is better and overfitting is controlled
                if val_acc > best_val_acc and train_test_gap < 15:
                    best_val_acc = val_acc
                    best_params = {
                        'C': C,
                        'max_features': max_features,
                        'min_df': min_df,
                        'model': model,
                        'vectorizer': vectorizer,
                        'results': results
                    }
    
    # Save tuning results
    with open('hyperparameter_tuning_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("BEST HYPERPARAMETERS")
    print("="*60)
    print(f"C: {best_params['C']}")
    print(f"max_features: {best_params['max_features']}")
    print(f"min_df: {best_params['min_df']}")
    print(f"Validation Accuracy: {best_val_acc:.4f}%")
    
    return best_params

def main(tune_hyperparameters=False):
    # Load data
    train_df, val_df, test_df = load_data()
    
    if tune_hyperparameters:
        # Run hyperparameter tuning
        best_params = hyperparameter_tuning(train_df, val_df, test_df)
        model = best_params['model']
        vectorizer = best_params['vectorizer']
        results = best_params['results']
    else:
        # Train with default regularized parameters
        model, vectorizer, results = train_tfidf_logreg(
            train_df, val_df, test_df,
            C=0.1,  # Strong regularization
            max_features=2000,  # Reduced features
            min_df=10,  # Filter rare terms
            max_df=0.8  # Filter very common terms
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Final summary
    if results['results']['improvements']['test'] > 0:
        print(f"\n✓ Model beats baseline by {results['results']['improvements']['test']:.4f}% on test set!")
    else:
        print(f"\n✗ Model underperforms baseline by {abs(results['results']['improvements']['test']):.4f}% on test set")
    
    print(f"Overfitting gap (train-test): {results['results']['overfitting_metrics']['train_test_gap']:.4f}%")

if __name__ == "__main__":
    main()