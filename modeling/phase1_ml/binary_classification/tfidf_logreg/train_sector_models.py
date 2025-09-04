import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load train, validation, and test datasets"""
    train_df = pd.read_csv('../../next_day_data/train.csv')
    val_df = pd.read_csv('../../next_day_data/val.csv')
    test_df = pd.read_csv('../../next_day_data/test.csv')
    
    print(f"Data loaded:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    return train_df, val_df, test_df

def analyze_sector_distribution(train_df, val_df, test_df):
    """Analyze sector and industry distribution in the data"""
    print("\n" + "="*70)
    print("SECTOR/INDUSTRY DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Combine all data for overall statistics
    all_df = pd.concat([train_df, val_df, test_df])
    
    # Sector distribution
    print("\n1. SECTOR DISTRIBUTION")
    print("-"*70)
    sector_counts = all_df['sector'].value_counts()
    print("\nTop 10 Sectors by sample count:")
    for sector, count in sector_counts.head(10).items():
        pct = count / len(all_df) * 100
        print(f"  {sector:<30} {count:5d} samples ({pct:5.2f}%)")
    
    # Industry distribution  
    print("\n2. INDUSTRY DISTRIBUTION")
    print("-"*70)
    industry_counts = all_df['industry'].value_counts()
    print("\nTop 15 Industries by sample count:")
    for industry, count in industry_counts.head(15).items():
        pct = count / len(all_df) * 100
        print(f"  {industry:<40} {count:5d} samples ({pct:5.2f}%)")
    
    # Sector-wise class balance
    print("\n3. SECTOR-WISE CLASS BALANCE (Binary)")
    print("-"*70)
    print(f"{'Sector':<30} {'Total':<10} {'UP %':<10} {'DOWN %':<10}")
    print("-"*60)
    
    for sector in sector_counts.head(10).index:
        sector_data = all_df[all_df['sector'] == sector]
        up_pct = (sector_data['binary_label'] == 'UP').mean() * 100
        down_pct = (sector_data['binary_label'] == 'DOWN').mean() * 100
        print(f"{sector:<30} {len(sector_data):<10} {up_pct:<10.2f} {down_pct:<10.2f}")
    
    return sector_counts, industry_counts

def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = ' '.join(text.split())
    return text

def train_sector_specific_model(train_df, val_df, test_df, sector, min_samples=100):
    """Train a model specific to a sector"""
    
    # Filter data for this sector
    train_sector = train_df[train_df['sector'] == sector].copy()
    val_sector = val_df[val_df['sector'] == sector].copy()
    test_sector = test_df[test_df['sector'] == sector].copy()
    
    # Check if we have enough samples
    if len(train_sector) < min_samples:
        return None
    
    # Preprocess text
    train_sector['processed_summary'] = train_sector['summary'].apply(preprocess_text)
    val_sector['processed_summary'] = val_sector['summary'].apply(preprocess_text)
    test_sector['processed_summary'] = test_sector['summary'].apply(preprocess_text)
    
    # Extract features and labels
    X_train = train_sector['processed_summary'].values
    y_train = train_sector['binary_target'].values
    X_val = val_sector['processed_summary'].values
    y_val = val_sector['binary_target'].values
    X_test = test_sector['processed_summary'].values
    y_test = test_sector['binary_target'].values
    
    # Check if test set has both classes
    if len(np.unique(y_test)) < 2 or len(test_sector) < 10:
        return None
    
    # Create TF-IDF vectorizer (adjusted for smaller data)
    max_features = min(2000, len(train_sector) // 2)
    min_df = max(2, min(5, len(train_sector) // 100))
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        stop_words='english'
    )
    
    try:
        # Fit and transform
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train_tfidf, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_tfidf)
        val_pred = model.predict(X_val_tfidf) if len(X_val) > 0 else []
        test_pred = model.predict(X_test_tfidf)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, train_pred) * 100
        val_acc = accuracy_score(y_val, val_pred) * 100 if len(y_val) > 0 else 0
        test_acc = accuracy_score(y_test, test_pred) * 100
        
        # Calculate baseline (majority class in test)
        test_majority = max(np.bincount(y_test)) / len(y_test) * 100
        
        return {
            'sector': sector,
            'train_samples': len(train_sector),
            'val_samples': len(val_sector),
            'test_samples': len(test_sector),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'test_baseline': test_majority,
            'improvement': test_acc - test_majority,
            'model': model,
            'vectorizer': vectorizer
        }
    except Exception as e:
        print(f"  Error training model for {sector}: {e}")
        return None

def train_general_model(train_df, val_df, test_df):
    """Train a general model on all data"""
    
    # Preprocess text
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
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        stop_words='english'
    )
    
    # Fit and transform
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver='liblinear',
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_tfidf)
    val_pred = model.predict(X_val_tfidf)
    test_pred = model.predict(X_test_tfidf)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_pred) * 100
    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100
    
    return model, vectorizer, train_acc, val_acc, test_acc

def evaluate_sector_models(train_df, val_df, test_df, sector_models, general_model, general_vectorizer):
    """Evaluate sector-specific models vs general model"""
    
    print("\n" + "="*100)
    print("SECTOR-SPECIFIC VS GENERAL MODEL COMPARISON")
    print("="*100)
    
    results = []
    
    # For each sector with a model
    for sector_result in sector_models:
        if sector_result is None:
            continue
            
        sector = sector_result['sector']
        
        # Get test data for this sector
        test_sector = test_df[test_df['sector'] == sector].copy()
        test_sector['processed_summary'] = test_sector['summary'].apply(preprocess_text)
        
        X_test = test_sector['processed_summary'].values
        y_test = test_sector['binary_target'].values
        
        # Evaluate general model on this sector
        X_test_general = general_vectorizer.transform(X_test)
        general_pred = general_model.predict(X_test_general)
        general_acc = accuracy_score(y_test, general_pred) * 100
        
        # Compare
        improvement = sector_result['test_acc'] - general_acc
        
        results.append({
            'sector': sector,
            'samples': sector_result['test_samples'],
            'baseline': sector_result['test_baseline'],
            'general_model': general_acc,
            'sector_model': sector_result['test_acc'],
            'improvement': improvement,
            'beats_baseline': sector_result['test_acc'] > sector_result['test_baseline'],
            'beats_general': sector_result['test_acc'] > general_acc
        })
    
    # Print results table
    print(f"\n{'Sector':<25} {'Samples':<10} {'Baseline':<12} {'General':<12} {'Sector':<12} {'Improve':<12} {'Status'}")
    print("-"*100)
    
    for r in sorted(results, key=lambda x: x['improvement'], reverse=True):
        status = ""
        if r['beats_baseline'] and r['beats_general']:
            status = "✓✓ Best"
        elif r['beats_baseline']:
            status = "✓ Beats Base"
        elif r['beats_general']:
            status = "✓ Beats Gen"
        else:
            status = "✗"
            
        print(f"{r['sector']:<25} {r['samples']:<10} {r['baseline']:<12.2f}% {r['general_model']:<12.2f}% "
              f"{r['sector_model']:<12.2f}% {r['improvement']:+11.2f}% {status}")
    
    return results

def main():
    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_data()
    
    # Analyze distribution
    sector_counts, industry_counts = analyze_sector_distribution(train_df, val_df, test_df)
    
    # Train general model first
    print("\n" + "="*70)
    print("TRAINING GENERAL MODEL")
    print("="*70)
    general_model, general_vectorizer, gen_train, gen_val, gen_test = train_general_model(
        train_df.copy(), val_df.copy(), test_df.copy()
    )
    print(f"\nGeneral Model Performance:")
    print(f"  Train: {gen_train:.2f}%")
    print(f"  Val:   {gen_val:.2f}%")
    print(f"  Test:  {gen_test:.2f}%")
    
    # Train sector-specific models for top sectors
    print("\n" + "="*70)
    print("TRAINING SECTOR-SPECIFIC MODELS")
    print("="*70)
    
    # Select top sectors with enough samples
    top_sectors = []
    for sector, count in sector_counts.items():
        train_count = len(train_df[train_df['sector'] == sector])
        if train_count >= 100:  # Minimum 100 training samples
            top_sectors.append(sector)
        if len(top_sectors) >= 15:  # Limit to top 15 sectors
            break
    
    print(f"\nTraining models for {len(top_sectors)} sectors with sufficient data...")
    
    sector_models = []
    for i, sector in enumerate(top_sectors, 1):
        print(f"\n{i}. Training model for: {sector}")
        result = train_sector_specific_model(train_df, val_df, test_df, sector)
        
        if result:
            print(f"   Train samples: {result['train_samples']}")
            print(f"   Test accuracy: {result['test_acc']:.2f}%")
            print(f"   Test baseline: {result['test_baseline']:.2f}%")
            print(f"   Improvement over baseline: {result['improvement']:+.2f}%")
            sector_models.append(result)
        else:
            print(f"   Skipped: Insufficient data")
    
    # Compare sector models with general model
    comparison_results = evaluate_sector_models(
        train_df, val_df, test_df, 
        sector_models, general_model, general_vectorizer
    )
    
    # Save results
    save_results = []
    for r in comparison_results:
        save_results.append({
            'sector': r['sector'],
            'test_samples': r['samples'],
            'baseline_acc': r['baseline'],
            'general_model_acc': r['general_model'],
            'sector_model_acc': r['sector_model'],
            'improvement_over_general': r['improvement'],
            'beats_baseline': bool(r['beats_baseline']),  # Convert numpy bool to Python bool
            'beats_general': bool(r['beats_general'])  # Convert numpy bool to Python bool
        })
    
    with open('sector_model_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Calculate summary statistics
    beats_baseline = sum(1 for r in comparison_results if r['beats_baseline'])
    beats_general = sum(1 for r in comparison_results if r['beats_general'])
    
    print(f"\nOut of {len(comparison_results)} sectors:")
    print(f"  {beats_baseline} sector models beat their baseline ({beats_baseline/len(comparison_results)*100:.1f}%)")
    print(f"  {beats_general} sector models beat the general model ({beats_general/len(comparison_results)*100:.1f}%)")
    
    # Find best performing sectors
    best_improvement = max(comparison_results, key=lambda x: x['improvement'])
    print(f"\nBest sector-specific improvement over general model:")
    print(f"  {best_improvement['sector']}: +{best_improvement['improvement']:.2f}% improvement")
    
    print("\nResults saved to: sector_model_results.json")

if __name__ == "__main__":
    main()