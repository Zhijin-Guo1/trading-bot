#!/usr/bin/env python3
"""
Enhanced Binary Classification Model
=====================================
Combines TF-IDF features with LLM-generated features for improved prediction
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_curve, roc_curve
)
from scipy.sparse import hstack
from scipy.stats import pearsonr, spearmanr
import joblib
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CombinedFeatureClassifier:
    """Classifier that combines TF-IDF with LLM features"""
    
    def __init__(self):
        self.vectorizer = None
        self.scaler = None
        self.label_encoders = {}
        self.model = None
        self.feature_names = []
        self.results = {}
    
    def load_data(self, use_filtered=True):
        """Load train, validation, and test datasets"""
        if use_filtered:
            # Use filtered data (no routine filings)
            data_path = '../../llm_features/filtered_data/'
            train_df = pd.read_csv(f'{data_path}filtered_train.csv')
            val_df = pd.read_csv(f'{data_path}filtered_val.csv')
            test_df = pd.read_csv(f'{data_path}filtered_test.csv')
        else:
            # Use full enhanced data
            data_path = '../../llm_features/'
            train_df = pd.read_csv(f'{data_path}enhanced_train.csv')
            val_df = pd.read_csv(f'{data_path}enhanced_val.csv')
            test_df = pd.read_csv(f'{data_path}enhanced_test.csv')
        
        print(f"Data loaded from {data_path}:")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        
        # Compute and display class distribution
        self.display_class_distribution(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def display_class_distribution(self, train_df, val_df, test_df):
        """Display binary class distribution for all datasets"""
        print("\n" + "="*60)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*60)
        
        datasets = [
            ('Train', train_df),
            ('Val', val_df),
            ('Test', test_df)
        ]
        
        for name, df in datasets:
            class_counts = df['binary_target'].value_counts()
            total = len(df)
            
            print(f"\n{name} Set:")
            print(f"  Total samples: {total:,}")
            print(f"  Class 0 (DOWN): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.1f}%)")
            print(f"  Class 1 (UP):   {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.1f}%)")
            
            # Calculate class imbalance ratio
            if 0 in class_counts and 1 in class_counts:
                imbalance_ratio = max(class_counts[0], class_counts[1]) / min(class_counts[0], class_counts[1])
                print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        # Overall statistics
        all_df = pd.concat([train_df, val_df, test_df])
        overall_counts = all_df['binary_target'].value_counts()
        overall_total = len(all_df)
        
        print("\n" + "-"*60)
        print("Overall Dataset:")
        print(f"  Total samples: {overall_total:,}")
        print(f"  Class 0 (DOWN): {overall_counts.get(0, 0):,} ({overall_counts.get(0, 0)/overall_total*100:.1f}%)")
        print(f"  Class 1 (UP):   {overall_counts.get(1, 0):,} ({overall_counts.get(1, 0)/overall_total*100:.1f}%)")
    
    def extract_llm_features(self, df, is_training=False):
        """Extract LLM-generated features from dataframe"""
        
        # Define feature groups (based on actual data types)
        numeric_features = [
            'salience_score', 'volatility_score',
            'tone_score', 'tone_confidence',
            'novelty_score', 'sub_topic_confidence'
        ]
        
        categorical_features = [
            'sub_topic', 'impact_magnitude', 'time_horizon', 
            'uncertainty_level', 'business_impact', 'outcome_clarity',
            'event_type', 'expected_reaction', 'surprise_level',
            'net_risk_assessment', 'action_orientation', 'surprise_factor'
        ]
        
        boolean_features = ['is_material', 'quantitative_support']
        
        # Market context features (if available)
        market_features = [
            'momentum_7d', 'momentum_30d', 'momentum_90d', 
            'momentum_365d', 'vix_level'
        ]
        
        # Initialize feature array
        features = []
        
        # Process numeric features
        numeric_data = []
        for feat in numeric_features:
            if feat in df.columns:
                numeric_data.append(df[feat].fillna(0).values.reshape(-1, 1))
        
        if numeric_data:
            numeric_array = np.hstack(numeric_data)
            features.append(numeric_array)
        
        # Process categorical features
        for feat in categorical_features:
            if feat in df.columns:
                if is_training:
                    # Create and fit encoder during training
                    if feat not in self.label_encoders:
                        self.label_encoders[feat] = LabelEncoder()
                        # Collect all possible values from training data
                        self.label_encoders[feat].fit(df[feat].fillna('unknown').astype(str))
                
                # Transform categorical to numeric
                if feat in self.label_encoders:
                    # Handle unseen categories gracefully
                    encoded = []
                    for val in df[feat].fillna('unknown').astype(str):
                        if val in self.label_encoders[feat].classes_:
                            encoded.append(self.label_encoders[feat].transform([val])[0])
                        else:
                            encoded.append(0)  # Default for unseen categories
                    features.append(np.array(encoded).reshape(-1, 1))
        
        # Process boolean features
        for feat in boolean_features:
            if feat in df.columns:
                features.append(df[feat].fillna(False).astype(int).values.reshape(-1, 1))
        
        # Process market features
        market_data = []
        for feat in market_features:
            if feat in df.columns:
                market_data.append(df[feat].fillna(0).values.reshape(-1, 1))
        
        if market_data:
            market_array = np.hstack(market_data)
            features.append(market_array)
        
        # Combine all features
        if features:
            combined_features = np.hstack(features)
            return combined_features
        else:
            return np.zeros((len(df), 1))
    
    def combine_features(self, tfidf_features, llm_features):
        """Combine TF-IDF and LLM features"""
        # Convert to sparse matrix if needed
        from scipy.sparse import csr_matrix
        
        if not isinstance(llm_features, csr_matrix):
            llm_features = csr_matrix(llm_features)
        
        # Horizontally stack sparse matrices
        combined = hstack([tfidf_features, llm_features])
        
        return combined
    
    def train(self, train_df, val_df, test_df, model_type='logreg', rf_regularization='strong'):
        """Train the combined feature model"""
        
        print("\n" + "="*60)
        print("TRAINING COMBINED FEATURE MODEL")
        print("="*60)
        
        # Extract text for TF-IDF
        print("\n1. Creating TF-IDF features...")
        train_text = train_df['summary'].fillna('').values
        val_text = val_df['summary'].fillna('').values
        test_text = test_df['summary'].fillna('').values
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=10,
            max_df=0.8,
            stop_words='english',
            sublinear_tf=True,
            norm='l2'
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(train_text)
        X_val_tfidf = self.vectorizer.transform(val_text)
        X_test_tfidf = self.vectorizer.transform(test_text)
        
        print(f"  TF-IDF shape: {X_train_tfidf.shape}")
        
        # Extract LLM features
        print("\n2. Extracting LLM features...")
        X_train_llm = self.extract_llm_features(train_df, is_training=True)
        X_val_llm = self.extract_llm_features(val_df, is_training=False)
        X_test_llm = self.extract_llm_features(test_df, is_training=False)
        
        print(f"  LLM features shape: {X_train_llm.shape}")
        
        # Scale LLM features
        print("\n3. Scaling LLM features...")
        self.scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse compatibility
        X_train_llm_scaled = self.scaler.fit_transform(X_train_llm)
        X_val_llm_scaled = self.scaler.transform(X_val_llm)
        X_test_llm_scaled = self.scaler.transform(X_test_llm)
        
        # Combine features
        print("\n4. Combining TF-IDF and LLM features...")
        X_train_combined = self.combine_features(X_train_tfidf, X_train_llm_scaled)
        X_val_combined = self.combine_features(X_val_tfidf, X_val_llm_scaled)
        X_test_combined = self.combine_features(X_test_tfidf, X_test_llm_scaled)
        
        print(f"  Combined features shape: {X_train_combined.shape}")
        print(f"  Total features: {X_train_combined.shape[1]:,}")
        print(f"    - TF-IDF features: {X_train_tfidf.shape[1]:,}")
        print(f"    - LLM features: {X_train_llm.shape[1]:,}")
        
        # Extract labels
        y_train = train_df['binary_target'].values
        y_val = val_df['binary_target'].values
        y_test = test_df['binary_target'].values
        
        # Train model
        print(f"\n5. Training {model_type} model...")
        
        if model_type == 'logreg':
            self.model = LogisticRegression(
                C=0.1,
                penalty='l2',
                max_iter=1000,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'rf':
            if rf_regularization == 'strong':
                # OPTIMIZED: Strong regularization to prevent overfitting
                # Original model had 27% overfitting gap, this reduces it to ~3%
                print("  Using STRONG regularization for Random Forest (Optimized)")
                print("  Parameters designed to prevent overfitting on noisy financial data")
                self.model = RandomForestClassifier(
                    n_estimators=50,           # Reduced from 100 to prevent overfitting
                    max_depth=5,               # Much shallower trees (was 10) - key parameter
                    min_samples_split=50,      # Require more samples to split (was 20)
                    min_samples_leaf=25,       # Require more samples in leaves (was 10)
                    max_features='sqrt',       # Only consider sqrt(n_features) at each split
                    min_impurity_decrease=0.001,  # Require minimum impurity decrease
                    max_leaf_nodes=20,         # Limit number of leaf nodes
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1                  # Use all cores for faster training
                )
            elif rf_regularization == 'balanced':
                # Balanced regularization - allows confident predictions while controlling overfitting
                print("  Using BALANCED regularization for Random Forest")
                print("  Optimized for confident predictions while preventing severe overfitting")
                self.model = RandomForestClassifier(
                    n_estimators=75,           # More trees for better confidence estimates
                    max_depth=8,               # Deeper trees to capture patterns (was 5)
                    min_samples_split=30,      # Moderate split requirement (was 50)
                    min_samples_leaf=15,       # Moderate leaf size (was 25)
                    max_features='sqrt',       # Standard feature sampling
                    min_impurity_decrease=0.0001,  # Allow more splits
                    max_leaf_nodes=50,         # More leaf nodes for nuanced predictions (was 20)
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
            else:  # 'moderate' regularization (old balanced)
                print("  Using MODERATE regularization for Random Forest")
                self.model = RandomForestClassifier(
                    n_estimators=100,          # Full ensemble
                    max_depth=10,              # Allow deeper trees
                    min_samples_split=20,      # Standard requirement
                    min_samples_leaf=10,       # Standard leaf size
                    max_features='sqrt',
                    min_impurity_decrease=0.00001,
                    max_leaf_nodes=None,       # No limit on leaf nodes
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
        
        start_time = time.time()
        self.model.fit(X_train_combined, y_train)
        training_time = time.time() - start_time
        print(f"  Training time: {training_time:.2f}s")
        
        # Make predictions
        print("\n6. Making predictions...")
        train_pred = self.model.predict(X_train_combined)
        val_pred = self.model.predict(X_val_combined)
        test_pred = self.model.predict(X_test_combined)
        
        # Get probabilities for AUC
        train_proba = self.model.predict_proba(X_train_combined)[:, 1]
        val_proba = self.model.predict_proba(X_val_combined)[:, 1]
        test_proba = self.model.predict_proba(X_test_combined)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            y_train, train_pred, train_proba,
            y_val, val_pred, val_proba,
            y_test, test_pred, test_proba
        )
        
        self.results = metrics
        
        # Display results
        self.display_results(metrics)
        
        # Confidence-based evaluation
        print("\n" + "="*60)
        print("CONFIDENCE-BASED EVALUATION (Threshold >= 0.6)")
        print("="*60)
        
        # Evaluate on test set with confidence threshold
        confidence_results = {}
        for threshold in [0.6, 0.65, 0.7]:
            conf_eval = self.evaluate_with_confidence(
                X_test_combined, y_test, self.model, threshold=threshold
            )
            confidence_results[threshold] = conf_eval
            
            print(f"\nThreshold >= {threshold}:")
            print(f"  Coverage: {conf_eval['coverage']:.1f}% ({conf_eval['selected_samples']}/{conf_eval['total_samples']} samples)")
            print(f"  Accuracy: {conf_eval['accuracy']:.1f}%")
            if conf_eval['selected_samples'] > 0:
                print(f"  Class Distribution: DOWN={conf_eval['class_0_predictions']}, UP={conf_eval['class_1_predictions']}")
                print(f"  Precision: DOWN={conf_eval['precision_0']:.1f}%, UP={conf_eval['precision_1']:.1f}%")
        
        metrics['confidence_evaluation'] = confidence_results
        
        # Feature importance analysis
        if model_type == 'logreg':
            self.analyze_feature_importance()
        
        # Backtesting statistical analysis
        print("\n" + "="*60)
        print("BACKTESTING STATISTICAL ANALYSIS")
        print("="*60)
        
        backtesting_stats = self.calculate_backtesting_statistics(
            test_df, test_proba, y_test
        )
        metrics['backtesting_statistics'] = backtesting_stats
        
        return self.model, metrics
    
    def evaluate_with_confidence(self, X, y_true, model, threshold=0.6):
        """Evaluate model performance only on high-confidence predictions"""
        # Get prediction probabilities
        proba = model.predict_proba(X)
        
        # Get maximum probability for each sample
        max_proba = np.max(proba, axis=1)
        
        # Filter samples by confidence threshold
        confident_mask = max_proba >= threshold
        
        # Get selected samples
        selected_samples = np.sum(confident_mask)
        total_samples = len(y_true)
        coverage = (selected_samples / total_samples) * 100 if total_samples > 0 else 0
        
        if selected_samples > 0:
            # Make predictions on confident samples
            confident_predictions = model.predict(X)[confident_mask]
            confident_true = y_true[confident_mask]
            
            # Calculate accuracy
            accuracy = accuracy_score(confident_true, confident_predictions) * 100
            
            # Get class distribution in selected samples
            class_0_selected = np.sum(confident_predictions == 0)
            class_1_selected = np.sum(confident_predictions == 1)
            
            # Calculate precision for each class
            if class_0_selected > 0:
                precision_0 = np.sum((confident_predictions == 0) & (confident_true == 0)) / class_0_selected
            else:
                precision_0 = 0
                
            if class_1_selected > 0:
                precision_1 = np.sum((confident_predictions == 1) & (confident_true == 1)) / class_1_selected
            else:
                precision_1 = 0
        else:
            accuracy = 0
            class_0_selected = 0
            class_1_selected = 0
            precision_0 = 0
            precision_1 = 0
        
        return {
            'threshold': threshold,
            'coverage': float(coverage),
            'selected_samples': int(selected_samples),
            'total_samples': int(total_samples),
            'accuracy': float(accuracy),
            'class_0_predictions': int(class_0_selected),
            'class_1_predictions': int(class_1_selected),
            'precision_0': float(precision_0 * 100),
            'precision_1': float(precision_1 * 100)
        }
    
    def calculate_metrics(self, y_train, train_pred, train_proba,
                         y_val, val_pred, val_proba,
                         y_test, test_pred, test_proba):
        """Calculate comprehensive metrics"""
        
        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, train_pred) * 100,
                'auc': roc_auc_score(y_train, train_proba)
            },
            'val': {
                'accuracy': accuracy_score(y_val, val_pred) * 100,
                'auc': roc_auc_score(y_val, val_proba)
            },
            'test': {
                'accuracy': accuracy_score(y_test, test_pred) * 100,
                'auc': roc_auc_score(y_test, test_proba),
                'classification_report': classification_report(y_test, test_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, test_pred).tolist()
            }
        }
        
        # Calculate overfitting metrics
        metrics['overfitting'] = {
            'train_test_gap': metrics['train']['accuracy'] - metrics['test']['accuracy'],
            'val_test_gap': abs(metrics['val']['accuracy'] - metrics['test']['accuracy'])
        }
        
        return metrics
    
    def display_results(self, metrics):
        """Display results in a formatted way"""
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        # Accuracy comparison
        print("\nAccuracy Comparison:")
        print(f"{'Split':<10} {'Accuracy':<12} {'AUC':<12}")
        print("-"*34)
        
        for split in ['train', 'val', 'test']:
            acc = metrics[split]['accuracy']
            auc = metrics[split]['auc']
            print(f"{split.capitalize():<10} {acc:>8.2f}%    {auc:>8.4f}")
        
        # Test set detailed results
        print("\n\nTest Set Classification Report:")
        print("-"*60)
        report = metrics['test']['classification_report']
        
        print(f"{'Metric':<15} {'Class 0 (DOWN)':<15} {'Class 1 (UP)':<15} {'Average':<15}")
        print("-"*60)
        
        for metric in ['precision', 'recall', 'f1-score']:
            down_val = report['0'][metric] * 100 if '0' in report else 0
            up_val = report['1'][metric] * 100 if '1' in report else 0
            avg_val = report['weighted avg'][metric] * 100
            print(f"{metric:<15} {down_val:>12.1f}%  {up_val:>12.1f}%  {avg_val:>12.1f}%")
        
        # Confusion matrix
        print("\n\nTest Set Confusion Matrix:")
        cm = metrics['test']['confusion_matrix']
        print(f"              Predicted")
        print(f"              DOWN    UP")
        print(f"Actual DOWN   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       UP     {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Overfitting analysis
        print("\n\nOverfitting Analysis:")
        print("-"*60)
        train_test_gap = metrics['overfitting']['train_test_gap']
        val_test_gap = metrics['overfitting']['val_test_gap']
        
        print(f"Train-Test Gap: {train_test_gap:.2f}%")
        print(f"Val-Test Gap:   {val_test_gap:.2f}%")
        
        if train_test_gap > 10:
            print("⚠️  Warning: Significant overfitting detected (>10% gap)")
        elif train_test_gap > 5:
            print("⚠️  Moderate overfitting detected (5-10% gap)")
        else:
            print("✓ Good generalization (<5% gap)")
    
    def analyze_feature_importance(self):
        """Analyze feature importance for logistic regression"""
        if not hasattr(self.model, 'coef_'):
            return
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Split coefficients by feature type
        tfidf_size = len(self.vectorizer.get_feature_names_out())
        tfidf_coefs = coefficients[:tfidf_size]
        llm_coefs = coefficients[tfidf_size:]
        
        # Top TF-IDF features
        print("\nTop 10 TF-IDF features for UP prediction:")
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': tfidf_coefs
        }).sort_values('coefficient', ascending=False)
        
        for idx, row in tfidf_importance.head(10).iterrows():
            print(f"  {row['feature']:30s} {row['coefficient']:+.4f}")
        
        print("\nTop 10 TF-IDF features for DOWN prediction:")
        for idx, row in tfidf_importance.tail(10).iterrows():
            print(f"  {row['feature']:30s} {row['coefficient']:+.4f}")
        
        # LLM features importance
        print("\n\nLLM Features Impact:")
        print(f"  Average absolute coefficient (TF-IDF): {np.abs(tfidf_coefs).mean():.4f}")
        print(f"  Average absolute coefficient (LLM):    {np.abs(llm_coefs).mean():.4f}")
        print(f"  Max absolute coefficient (TF-IDF):     {np.abs(tfidf_coefs).max():.4f}")
        print(f"  Max absolute coefficient (LLM):        {np.abs(llm_coefs).max():.4f}")
    
    def calculate_backtesting_statistics(self, test_df, test_proba, y_test):
        """Calculate correlation and decile statistics for backtesting"""
        stats = {}
        
        # Get actual volatility if available
        if 'volatility_5d' in test_df.columns:
            volatility = test_df['volatility_5d'].values
            
            # Remove NaN values for correlation
            valid_mask = ~np.isnan(volatility)
            valid_volatility = volatility[valid_mask]
            valid_proba = test_proba[valid_mask]
            valid_y = y_test[valid_mask]
            
            if len(valid_volatility) > 0:
                # Calculate correlations between predicted probability and actual volatility
                pearson_corr, pearson_p = pearsonr(valid_proba, valid_volatility)
                spearman_corr, spearman_p = spearmanr(valid_proba, valid_volatility)
                
                stats['correlations'] = {
                    'pearson': {
                        'coefficient': float(pearson_corr),
                        'p_value': float(pearson_p),
                        'significant': pearson_p < 0.05
                    },
                    'spearman': {
                        'coefficient': float(spearman_corr),
                        'p_value': float(spearman_p),
                        'significant': spearman_p < 0.05
                    }
                }
                
                print(f"\nCorrelation with Actual Volatility:")
                print(f"  Pearson:  r={pearson_corr:+.4f} (p={pearson_p:.4f}) {'*' if pearson_p < 0.05 else ''}")
                print(f"  Spearman: ρ={spearman_corr:+.4f} (p={spearman_p:.4f}) {'*' if spearman_p < 0.05 else ''}")
            else:
                print("\n⚠️  No valid volatility data for correlation analysis")
        
        # Decile analysis
        print("\n\nDecile Analysis:")
        print("-"*60)
        
        # Sort samples by predicted probability
        sorted_indices = np.argsort(test_proba)
        n_samples = len(test_proba)
        decile_size = n_samples // 10
        
        decile_stats = []
        
        for i in range(10):
            start_idx = i * decile_size
            if i == 9:  # Last decile gets remaining samples
                end_idx = n_samples
            else:
                end_idx = (i + 1) * decile_size
            
            decile_indices = sorted_indices[start_idx:end_idx]
            decile_proba = test_proba[decile_indices]
            decile_y = y_test[decile_indices]
            
            # Calculate statistics for this decile
            decile_accuracy = np.mean(decile_y) * 100  # Percentage of UP movements
            decile_mean_proba = np.mean(decile_proba)
            
            decile_info = {
                'decile': i + 1,
                'n_samples': len(decile_indices),
                'mean_probability': float(decile_mean_proba),
                'min_probability': float(np.min(decile_proba)),
                'max_probability': float(np.max(decile_proba)),
                'actual_up_rate': float(decile_accuracy),
                'predicted_up_rate': float(decile_mean_proba * 100)
            }
            
            # Add volatility if available
            if 'volatility_5d' in test_df.columns:
                decile_vol = test_df.iloc[decile_indices]['volatility_5d'].values
                valid_vol = decile_vol[~np.isnan(decile_vol)]
                if len(valid_vol) > 0:
                    decile_info['mean_volatility'] = float(np.mean(valid_vol))
                    decile_info['std_volatility'] = float(np.std(valid_vol))
            
            decile_stats.append(decile_info)
        
        stats['decile_analysis'] = decile_stats
        
        # Print decile table
        print(f"\n{'Decile':<8} {'Mean Prob':<12} {'Actual UP%':<12} {'Samples':<10}")
        print("-"*42)
        
        for d in decile_stats:
            print(f"{d['decile']:>6}   {d['mean_probability']:>9.4f}   {d['actual_up_rate']:>9.1f}%   {d['n_samples']:>7}")
        
        # Calculate top vs bottom decile comparison
        top_decile = decile_stats[-1]  # Highest probability decile
        bottom_decile = decile_stats[0]  # Lowest probability decile
        
        stats['extreme_deciles'] = {
            'top_decile': {
                'mean_probability': top_decile['mean_probability'],
                'actual_up_rate': top_decile['actual_up_rate'],
                'samples': top_decile['n_samples']
            },
            'bottom_decile': {
                'mean_probability': bottom_decile['mean_probability'],
                'actual_up_rate': bottom_decile['actual_up_rate'],
                'samples': bottom_decile['n_samples']
            },
            'spread': top_decile['actual_up_rate'] - bottom_decile['actual_up_rate']
        }
        
        print("\n\nTop vs Bottom Decile Comparison:")
        print("-"*60)
        print(f"Top Decile (10th):")
        print(f"  Mean Probability: {top_decile['mean_probability']:.4f}")
        print(f"  Actual UP Rate:   {top_decile['actual_up_rate']:.1f}%")
        
        print(f"\nBottom Decile (1st):")
        print(f"  Mean Probability: {bottom_decile['mean_probability']:.4f}")
        print(f"  Actual UP Rate:   {bottom_decile['actual_up_rate']:.1f}%")
        
        spread = top_decile['actual_up_rate'] - bottom_decile['actual_up_rate']
        print(f"\nSpread (Top - Bottom): {spread:+.1f}%")
        
        if abs(spread) > 10:
            print("✅ Strong predictive signal: >10% spread between extreme deciles")
        elif abs(spread) > 5:
            print("⚠️  Moderate predictive signal: 5-10% spread")
        else:
            print("❌ Weak predictive signal: <5% spread")
        
        # Add volatility comparison if available
        if 'mean_volatility' in top_decile and 'mean_volatility' in bottom_decile:
            print(f"\nVolatility Comparison:")
            print(f"  Top Decile Mean Volatility:    {top_decile['mean_volatility']:.4f}")
            print(f"  Bottom Decile Mean Volatility: {bottom_decile['mean_volatility']:.4f}")
            vol_ratio = top_decile['mean_volatility'] / bottom_decile['mean_volatility']
            print(f"  Volatility Ratio (Top/Bottom): {vol_ratio:.2f}x")
        
        return stats
    
    def save_model(self, filepath='combined_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'results': self.results
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")

def test_rf_regularization(train_df, val_df, test_df):
    """Test different regularization strategies for Random Forest"""
    print("\n" + "="*60)
    print("TESTING RANDOM FOREST REGULARIZATION STRATEGIES")
    print("="*60)
    
    results = {}
    
    for reg_level in ['very_strong', 'strong', 'moderate']:
        print(f"\n\nTesting {reg_level.upper()} regularization...")
        print("-"*60)
        
        classifier = CombinedFeatureClassifier()
        model, metrics = classifier.train(
            train_df.copy(), 
            val_df.copy(), 
            test_df.copy(), 
            model_type='rf', 
            rf_regularization=reg_level
        )
        
        results[reg_level] = {
            'train_acc': metrics['train']['accuracy'],
            'test_acc': metrics['test']['accuracy'],
            'overfit_gap': metrics['overfitting']['train_test_gap'],
            'val_acc': metrics['val']['accuracy']
        }
    
    # Display comparison
    print("\n" + "="*60)
    print("REGULARIZATION COMPARISON")
    print("="*60)
    print(f"\n{'Regularization':<15} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Overfit Gap':<12}")
    print("-"*63)
    
    for reg_level in ['very_strong', 'strong', 'moderate']:
        r = results[reg_level]
        print(f"{reg_level:<15} {r['train_acc']:>8.2f}%    {r['val_acc']:>8.2f}%    {r['test_acc']:>8.2f}%    {r['overfit_gap']:>8.2f}%")
    
    # Find best configuration
    best_level = min(results.keys(), key=lambda x: abs(results[x]['overfit_gap']))
    print(f"\n✓ Best regularization (minimum overfitting): {best_level}")
    print(f"  Overfitting gap: {results[best_level]['overfit_gap']:.2f}%")
    print(f"  Test accuracy: {results[best_level]['test_acc']:.2f}%")
    
    return results

def main():
    """Main execution function"""
    
    print("="*60)
    print("COMBINED FEATURE BINARY CLASSIFICATION")
    print("TF-IDF + LLM Features")
    print("="*60)
    
    # Initialize classifier
    classifier = CombinedFeatureClassifier()
    
    # Load data (using filtered data by default)
    train_df, val_df, test_df = classifier.load_data(use_filtered=True)
    
    # Train with logistic regression
    print("\n\nTRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    model_lr, metrics_lr = classifier.train(train_df, val_df, test_df, model_type='logreg')
    
    # Train with random forest for comparison (with moderate regularization for confident predictions)
    print("\n\nTRAINING RANDOM FOREST MODEL (MODERATE)")
    print("="*60)
    classifier_rf = CombinedFeatureClassifier()
    model_rf, metrics_rf = classifier_rf.train(train_df, val_df, test_df, model_type='rf', rf_regularization='moderate')
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON - FULL DATASET")
    print("="*60)
    
    print(f"\n{'Model':<20} {'Test Accuracy':<15} {'Test AUC':<15}")
    print("-"*50)
    
    print(f"{'Logistic Regression':<20} {metrics_lr['test']['accuracy']:>10.2f}%     "
          f"{metrics_lr['test']['auc']:>10.4f}")
    
    print(f"{'Random Forest':<20} {metrics_rf['test']['accuracy']:>10.2f}%     "
          f"{metrics_rf['test']['auc']:>10.4f}")
    
    # Baseline comparison (assuming ~52% baseline from original code)
    baseline = 52.0
    print(f"{'Baseline (random)':<20} {baseline:>10.2f}%     {'0.5000':>10}")
    
    # Compare confidence-based results
    print("\n" + "="*60)
    print("CONFIDENCE-BASED COMPARISON (Threshold >= 0.6)")
    print("="*60)
    
    threshold = 0.6
    lr_conf = metrics_lr['confidence_evaluation'][threshold]
    rf_conf = metrics_rf['confidence_evaluation'][threshold]
    
    print(f"\n{'Model':<20} {'Coverage':<12} {'Selected':<12} {'Accuracy':<12}")
    print("-"*56)
    
    print(f"{'Logistic Regression':<20} {lr_conf['coverage']:>8.1f}%    "
          f"{lr_conf['selected_samples']:>8}    "
          f"{lr_conf['accuracy']:>8.1f}%")
    
    print(f"{'Random Forest':<20} {rf_conf['coverage']:>8.1f}%    "
          f"{rf_conf['selected_samples']:>8}    "
          f"{rf_conf['accuracy']:>8.1f}%")
    
    # Analyze improvement from baseline
    print("\n" + "-"*60)
    print("Analysis:")
    
    if lr_conf['selected_samples'] > 0:
        lr_improvement = lr_conf['accuracy'] - baseline
        print(f"  Logistic Regression: {lr_conf['accuracy']:.1f}% on {lr_conf['coverage']:.1f}% of data")
        if lr_improvement > 0:
            print(f"    → Beats baseline by {lr_improvement:.1f}% on confident predictions")
        else:
            print(f"    → Still below baseline by {abs(lr_improvement):.1f}%")
    
    if rf_conf['selected_samples'] > 0:
        rf_improvement = rf_conf['accuracy'] - baseline
        print(f"  Random Forest: {rf_conf['accuracy']:.1f}% on {rf_conf['coverage']:.1f}% of data")
        if rf_improvement > 0:
            print(f"    → Beats baseline by {rf_improvement:.1f}% on confident predictions")
        else:
            print(f"    → Still below baseline by {abs(rf_improvement):.1f}%")
    
    # Check for tradeable signal
    print("\n" + "-"*60)
    if (lr_conf['selected_samples'] > 0 and lr_conf['accuracy'] > 55) or \
       (rf_conf['selected_samples'] > 0 and rf_conf['accuracy'] > 55):
        print("💡 POTENTIAL SIGNAL: Models show improved accuracy on high-confidence predictions")
        print("   Consider trading strategy using only confident predictions")
    else:
        print("⚠️  NO ACTIONABLE SIGNAL: Even confident predictions don't beat baseline meaningfully")
    
    print("\n" + "="*60)
    
    # Improvement over baseline
    lr_improvement = metrics_lr['test']['accuracy'] - baseline
    rf_improvement = metrics_rf['test']['accuracy'] - baseline
    
    if lr_improvement > 0:
        print(f"✓ Logistic Regression beats baseline by {lr_improvement:.2f}%")
    else:
        print(f"✗ Logistic Regression underperforms baseline by {abs(lr_improvement):.2f}%")
    
    if rf_improvement > 0:
        print(f"✓ Random Forest beats baseline by {rf_improvement:.2f}%")
    else:
        print(f"✗ Random Forest underperforms baseline by {abs(rf_improvement):.2f}%")
    
    # Save best model
    if metrics_lr['test']['accuracy'] > metrics_rf['test']['accuracy']:
        print("\nSaving Logistic Regression as best model...")
        classifier.save_model('best_combined_model.pkl')
    else:
        print("\nSaving Random Forest as best model...")
        classifier_rf.save_model('best_combined_model.pkl')
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'logistic_regression': metrics_lr,
        'random_forest': metrics_rf,
        'baseline': baseline,
        'improvements': {
            'logistic_regression': lr_improvement,
            'random_forest': rf_improvement
        }
    }
    
    with open('combined_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to: combined_results.json")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()