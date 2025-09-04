#!/usr/bin/env python3
"""
Sector-based Analysis for Binary Classification
================================================
1. Train one model, evaluate per sector
2. Train sector-specific models
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import json
from datetime import datetime

class SectorAnalyzer:
    def __init__(self):
        self.results = {}
        self.sector_models = {}
        self.sector_vectorizers = {}
        self.sector_scalers = {}
        self.sector_label_encoders = {}
        
    def load_data(self):
        """Load train, validation, and test datasets"""
        print("Loading data...")
        
        # Load filtered data
        data_path = '../../llm_features/filtered_data/'
        train_df = pd.read_csv(f'{data_path}filtered_train.csv')
        val_df = pd.read_csv(f'{data_path}filtered_val.csv')
        test_df = pd.read_csv(f'{data_path}filtered_test.csv')
        
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def analyze_sector_distribution(self, train_df, test_df):
        """Analyze sector distribution and baselines"""
        print("\n" + "="*70)
        print("SECTOR DISTRIBUTION ANALYSIS")
        print("="*70)
        
        sectors = train_df['sector'].value_counts()
        test_sectors = test_df['sector'].value_counts()
        
        print(f"\n{'Sector':<25} {'Train':<10} {'Test':<10} {'Test %':<10}")
        print("-"*55)
        
        for sector in sectors.index:
            train_count = sectors.get(sector, 0)
            test_count = test_sectors.get(sector, 0) if sector in test_sectors.index else 0
            test_pct = (test_count / len(test_df)) * 100
            print(f"{sector[:25]:<25} {train_count:<10} {test_count:<10} {test_pct:<10.1f}%")
        
        # Calculate sector-specific baselines
        print("\n" + "="*70)
        print("SECTOR-SPECIFIC BASELINES (Test Set)")
        print("="*70)
        
        sector_baselines = {}
        print(f"\n{'Sector':<25} {'Total':<10} {'UP %':<10} {'DOWN %':<10} {'Baseline':<10}")
        print("-"*65)
        
        for sector in test_df['sector'].unique():
            # Handle NaN or non-string sectors
            if pd.isna(sector) or not isinstance(sector, str):
                continue
                
            sector_data = test_df[test_df['sector'] == sector]
            total = len(sector_data)
            up_pct = (sector_data['binary_target'].sum() / total) * 100
            down_pct = 100 - up_pct
            baseline = max(up_pct, down_pct)
            sector_baselines[sector] = baseline
            
            sector_name = str(sector)[:25]  # Convert to string and truncate
            print(f"{sector_name:<25} {total:<10} {up_pct:<10.1f}% {down_pct:<10.1f}% {baseline:<10.1f}%")
        
        # Overall baseline
        overall_up = (test_df['binary_target'].sum() / len(test_df)) * 100
        overall_baseline = max(overall_up, 100 - overall_up)
        print(f"\n{'OVERALL':<25} {len(test_df):<10} {overall_up:<10.1f}% {100-overall_up:<10.1f}% {overall_baseline:<10.1f}%")
        
        sector_baselines['OVERALL'] = overall_baseline
        return sector_baselines
    
    def extract_features(self, df, vectorizer=None, scaler=None, label_encoders=None, fit=False):
        """Extract TF-IDF and LLM features"""
        # TF-IDF features
        text = df['summary'].fillna('').values
        
        if fit:
            vectorizer = TfidfVectorizer(
                max_features=1000,  # Reduced for sector-specific models
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.9,
                stop_words='english',
                sublinear_tf=True
            )
            X_tfidf = vectorizer.fit_transform(text)
        else:
            X_tfidf = vectorizer.transform(text)
        
        # LLM features
        numeric_features = [
            'salience_score', 'volatility_score', 'tone_score',
            'tone_confidence', 'novelty_score', 'sub_topic_confidence'
        ]
        
        categorical_features = [
            'sub_topic', 'impact_magnitude', 'time_horizon',
            'uncertainty_level', 'business_impact'
        ]
        
        # Extract numeric features
        features = []
        for feat in numeric_features:
            if feat in df.columns:
                features.append(df[feat].fillna(0).values.reshape(-1, 1))
        
        # Process categorical features
        if fit:
            label_encoders = {}
            for feat in categorical_features:
                if feat in df.columns:
                    le = LabelEncoder()
                    le.fit(df[feat].fillna('unknown').astype(str))
                    label_encoders[feat] = le
                    encoded = le.transform(df[feat].fillna('unknown').astype(str))
                    features.append(encoded.reshape(-1, 1))
        else:
            for feat in categorical_features:
                if feat in df.columns and feat in label_encoders:
                    le = label_encoders[feat]
                    encoded = []
                    for val in df[feat].fillna('unknown').astype(str):
                        if val in le.classes_:
                            encoded.append(le.transform([val])[0])
                        else:
                            encoded.append(0)
                    features.append(np.array(encoded).reshape(-1, 1))
        
        # Combine and scale
        if features:
            X_llm = np.hstack(features)
            if fit:
                scaler = StandardScaler(with_mean=False)
                X_llm_scaled = scaler.fit_transform(X_llm)
            else:
                X_llm_scaled = scaler.transform(X_llm)
        else:
            X_llm_scaled = np.zeros((len(df), 1))
        
        # Combine TF-IDF and LLM features
        X_combined = hstack([X_tfidf, csr_matrix(X_llm_scaled)])
        
        return X_combined, vectorizer, scaler, label_encoders
    
    def train_global_model(self, train_df, val_df, test_df):
        """Train one model on all data"""
        print("\n" + "="*70)
        print("TRAINING GLOBAL MODEL (All Sectors)")
        print("="*70)
        
        # Extract features
        X_train, vectorizer, scaler, label_encoders = self.extract_features(
            train_df, fit=True
        )
        X_val, _, _, _ = self.extract_features(
            val_df, vectorizer, scaler, label_encoders, fit=False
        )
        X_test, _, _, _ = self.extract_features(
            test_df, vectorizer, scaler, label_encoders, fit=False
        )
        
        # Extract labels
        y_train = train_df['binary_target'].values
        y_val = val_df['binary_target'].values
        y_test = test_df['binary_target'].values
        
        # Train model
        model = LogisticRegression(
            C=0.1,
            penalty='l2',
            max_iter=1000,
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
        
        print("Training...")
        model.fit(X_train, y_train)
        
        # Overall performance
        train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
        val_acc = accuracy_score(y_val, model.predict(X_val)) * 100
        test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
        
        print(f"\nOverall Performance:")
        print(f"  Train: {train_acc:.2f}%")
        print(f"  Val:   {val_acc:.2f}%")
        print(f"  Test:  {test_acc:.2f}%")
        
        return model, vectorizer, scaler, label_encoders, test_acc
    
    def evaluate_global_model_by_sector(self, model, test_df, vectorizer, scaler, label_encoders, sector_baselines):
        """Evaluate global model on each sector"""
        print("\n" + "="*70)
        print("GLOBAL MODEL - SECTOR-SPECIFIC EVALUATION")
        print("="*70)
        
        results = {}
        
        print(f"\n{'Sector':<25} {'Samples':<10} {'Accuracy':<12} {'Baseline':<12} {'vs Baseline':<12} {'vs 52%':<12}")
        print("-"*93)
        
        # Filter out NaN sectors
        valid_sectors = [s for s in test_df['sector'].unique() if pd.notna(s) and isinstance(s, str)]
        
        for sector in sorted(valid_sectors):
            # Get sector data
            sector_data = test_df[test_df['sector'] == sector]
            if len(sector_data) < 10:  # Skip sectors with too few samples
                continue
            
            # Extract features
            X_sector, _, _, _ = self.extract_features(
                sector_data, vectorizer, scaler, label_encoders, fit=False
            )
            y_sector = sector_data['binary_target'].values
            
            # Predict
            y_pred = model.predict(X_sector)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_sector, y_pred) * 100
            
            # Compare with baselines
            sector_baseline = sector_baselines.get(sector, 52)
            vs_baseline = accuracy - sector_baseline
            vs_52 = accuracy - 52
            
            results[sector] = {
                'samples': len(sector_data),
                'accuracy': accuracy,
                'baseline': sector_baseline,
                'vs_baseline': vs_baseline,
                'vs_52': vs_52
            }
            
            # Format output
            baseline_symbol = "✓" if vs_baseline > 0 else "✗"
            overall_symbol = "✓" if vs_52 > 0 else "✗"
            
            print(f"{sector[:25]:<25} {len(sector_data):<10} {accuracy:<12.2f}% {sector_baseline:<12.2f}% "
                  f"{vs_baseline:+11.2f}% {baseline_symbol} {vs_52:+11.2f}% {overall_symbol}")
        
        # Summary statistics
        accuracies = [r['accuracy'] for r in results.values()]
        beats_baseline = sum(1 for r in results.values() if r['vs_baseline'] > 0)
        beats_52 = sum(1 for r in results.values() if r['vs_52'] > 0)
        
        print("\n" + "-"*93)
        print(f"Summary:")
        print(f"  Average sector accuracy: {np.mean(accuracies):.2f}%")
        print(f"  Best sector: {max(results.items(), key=lambda x: x[1]['accuracy'])[0]} ({max(accuracies):.2f}%)")
        print(f"  Worst sector: {min(results.items(), key=lambda x: x[1]['accuracy'])[0]} ({min(accuracies):.2f}%)")
        print(f"  Sectors beating their baseline: {beats_baseline}/{len(results)}")
        print(f"  Sectors beating 52%: {beats_52}/{len(results)}")
        
        return results
    
    def train_sector_specific_models(self, train_df, val_df, test_df, sector_baselines):
        """Train separate model for each sector"""
        print("\n" + "="*70)
        print("TRAINING SECTOR-SPECIFIC MODELS")
        print("="*70)
        
        results = {}
        
        print(f"\n{'Sector':<25} {'Train':<8} {'Test':<8} {'Accuracy':<12} {'Baseline':<12} {'vs Baseline':<12}")
        print("-"*81)
        
        # Filter out NaN sectors
        valid_sectors = [s for s in train_df['sector'].unique() if pd.notna(s) and isinstance(s, str)]
        
        for sector in sorted(valid_sectors):
            # Get sector-specific data
            train_sector = train_df[train_df['sector'] == sector]
            val_sector = val_df[val_df['sector'] == sector]
            test_sector = test_df[test_df['sector'] == sector]
            
            # Skip if too few samples
            if len(train_sector) < 50 or len(test_sector) < 10:
                continue
            
            # Extract features for this sector
            X_train, vectorizer, scaler, label_encoders = self.extract_features(
                train_sector, fit=True
            )
            X_val, _, _, _ = self.extract_features(
                val_sector, vectorizer, scaler, label_encoders, fit=False
            )
            X_test, _, _, _ = self.extract_features(
                test_sector, vectorizer, scaler, label_encoders, fit=False
            )
            
            y_train = train_sector['binary_target'].values
            y_val = val_sector['binary_target'].values
            y_test = test_sector['binary_target'].values
            
            # Train sector-specific model
            model = LogisticRegression(
                C=0.1,
                penalty='l2',
                max_iter=1000,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            )
            
            try:
                model.fit(X_train, y_train)
                
                # Evaluate
                train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
                test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
                
                # Compare with baseline
                sector_baseline = sector_baselines.get(sector, 52)
                vs_baseline = test_acc - sector_baseline
                
                results[sector] = {
                    'train_samples': len(train_sector),
                    'test_samples': len(test_sector),
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'baseline': sector_baseline,
                    'vs_baseline': vs_baseline,
                    'overfitting': train_acc - test_acc
                }
                
                # Store model
                self.sector_models[sector] = model
                self.sector_vectorizers[sector] = vectorizer
                self.sector_scalers[sector] = scaler
                self.sector_label_encoders[sector] = label_encoders
                
                # Format output
                symbol = "✓" if vs_baseline > 0 else "✗"
                print(f"{sector[:25]:<25} {len(train_sector):<8} {len(test_sector):<8} "
                      f"{test_acc:<12.2f}% {sector_baseline:<12.2f}% {vs_baseline:+11.2f}% {symbol}")
                
            except Exception as e:
                print(f"{sector[:25]:<25} Failed: {str(e)[:40]}")
        
        # Summary
        if results:
            accuracies = [r['test_accuracy'] for r in results.values()]
            beats_baseline = sum(1 for r in results.values() if r['vs_baseline'] > 0)
            
            print("\n" + "-"*81)
            print(f"Summary:")
            print(f"  Models trained: {len(results)}")
            print(f"  Average accuracy: {np.mean(accuracies):.2f}%")
            print(f"  Best sector: {max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]} ({max(accuracies):.2f}%)")
            print(f"  Worst sector: {min(results.items(), key=lambda x: x[1]['test_accuracy'])[0]} ({min(accuracies):.2f}%)")
            print(f"  Sectors beating baseline: {beats_baseline}/{len(results)}")
        
        return results
    
    def compare_approaches(self, global_results, sector_results):
        """Compare global vs sector-specific models"""
        print("\n" + "="*70)
        print("COMPARISON: GLOBAL MODEL vs SECTOR-SPECIFIC MODELS")
        print("="*70)
        
        print(f"\n{'Sector':<25} {'Global Model':<15} {'Sector Model':<15} {'Winner':<12} {'Difference':<12}")
        print("-"*79)
        
        comparison = {}
        
        for sector in sorted(set(global_results.keys()) & set(sector_results.keys())):
            global_acc = global_results[sector]['accuracy']
            sector_acc = sector_results[sector]['test_accuracy']
            diff = sector_acc - global_acc
            
            winner = "Sector" if diff > 0 else "Global"
            
            comparison[sector] = {
                'global_accuracy': global_acc,
                'sector_accuracy': sector_acc,
                'difference': diff,
                'winner': winner
            }
            
            print(f"{sector[:25]:<25} {global_acc:<15.2f}% {sector_acc:<15.2f}% "
                  f"{winner:<12} {diff:+11.2f}%")
        
        # Summary
        sector_wins = sum(1 for c in comparison.values() if c['winner'] == 'Sector')
        global_wins = len(comparison) - sector_wins
        avg_diff = np.mean([c['difference'] for c in comparison.values()])
        
        print("\n" + "-"*79)
        print(f"Summary:")
        print(f"  Sector-specific models win: {sector_wins}/{len(comparison)} sectors")
        print(f"  Global model wins: {global_wins}/{len(comparison)} sectors")
        print(f"  Average difference: {avg_diff:+.2f}% (positive = sector model better)")
        
        return comparison
    
    def identify_predictable_sectors(self, global_results, sector_results, sector_baselines):
        """Identify which sectors are most predictable"""
        print("\n" + "="*70)
        print("MOST PREDICTABLE SECTORS")
        print("="*70)
        
        # Combine results
        predictability = []
        
        for sector in global_results.keys():
            baseline = sector_baselines.get(sector, 52)
            global_acc = global_results[sector]['accuracy']
            sector_acc = sector_results.get(sector, {}).get('test_accuracy', 0)
            
            best_acc = max(global_acc, sector_acc)
            edge_over_baseline = best_acc - baseline
            edge_over_52 = best_acc - 52
            
            predictability.append({
                'sector': sector,
                'best_accuracy': best_acc,
                'baseline': baseline,
                'edge_over_baseline': edge_over_baseline,
                'edge_over_52': edge_over_52,
                'best_model': 'sector' if sector_acc > global_acc else 'global'
            })
        
        # Sort by edge over baseline
        predictability.sort(key=lambda x: x['edge_over_baseline'], reverse=True)
        
        print(f"\n{'Rank':<6} {'Sector':<25} {'Best Acc':<10} {'Baseline':<10} {'Edge':<10} {'Model':<10}")
        print("-"*71)
        
        for i, p in enumerate(predictability[:10], 1):  # Top 10
            symbol = "✓" if p['edge_over_baseline'] > 0 else "✗"
            print(f"{i:<6} {p['sector'][:25]:<25} {p['best_accuracy']:<10.2f}% "
                  f"{p['baseline']:<10.2f}% {p['edge_over_baseline']:+9.2f}% {symbol} {p['best_model']:<10}")
        
        # Identify consistently predictable sectors
        good_sectors = [p for p in predictability if p['edge_over_baseline'] > 2]  # >2% edge
        
        if good_sectors:
            print(f"\n💡 TRADEABLE SECTORS (>2% edge over baseline):")
            for p in good_sectors:
                print(f"  • {p['sector']}: {p['best_accuracy']:.2f}% accuracy ({p['edge_over_baseline']:+.2f}% edge)")
        else:
            print("\n⚠️  No sectors show consistent edge > 2% over baseline")
        
        return predictability

def main():
    print("="*70)
    print("SECTOR-BASED BINARY CLASSIFICATION ANALYSIS")
    print("="*70)
    
    analyzer = SectorAnalyzer()
    
    # Load data
    train_df, val_df, test_df = analyzer.load_data()
    
    # Analyze sector distribution and baselines
    sector_baselines = analyzer.analyze_sector_distribution(train_df, test_df)
    
    # Approach 1: Train global model, evaluate by sector
    print("\n" + "="*70)
    print("APPROACH 1: GLOBAL MODEL WITH SECTOR EVALUATION")
    print("="*70)
    
    global_model, vectorizer, scaler, label_encoders, overall_acc = analyzer.train_global_model(
        train_df, val_df, test_df
    )
    
    global_results = analyzer.evaluate_global_model_by_sector(
        global_model, test_df, vectorizer, scaler, label_encoders, sector_baselines
    )
    
    # Approach 2: Train sector-specific models
    print("\n" + "="*70)
    print("APPROACH 2: SECTOR-SPECIFIC MODELS")
    print("="*70)
    
    sector_results = analyzer.train_sector_specific_models(
        train_df, val_df, test_df, sector_baselines
    )
    
    # Compare approaches
    comparison = analyzer.compare_approaches(global_results, sector_results)
    
    # Identify predictable sectors
    predictability = analyzer.identify_predictable_sectors(
        global_results, sector_results, sector_baselines
    )
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall_baseline': sector_baselines.get('OVERALL', 52),
        'global_model_accuracy': overall_acc,
        'sector_baselines': sector_baselines,
        'global_results_by_sector': global_results,
        'sector_specific_results': sector_results,
        'comparison': comparison,
        'predictability_ranking': predictability[:10]  # Top 10
    }
    
    with open('sector_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print("\n" + "="*70)
    print("Results saved to: sector_analysis_results.json")
    print("="*70)

if __name__ == "__main__":
    main()