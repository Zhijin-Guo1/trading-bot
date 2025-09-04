#!/usr/bin/env python3
"""
Combined Features Regression for Volatility Prediction
========================================================
Combines TF-IDF text features with LLM-generated features to predict
stock volatility (absolute returns) and raw returns.

Key hypothesis: Text features correlate better with volatility magnitude
than with directional returns.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import hstack, csr_matrix
import joblib
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CombinedVolatilityRegressor:
    """Regression model combining TF-IDF with LLM features for volatility prediction"""
    
    def __init__(self):
        self.vectorizer = None
        self.scaler = None
        self.feature_scaler = None
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.feature_names = []
    
    def load_data(self, use_filtered=True):
        """Load train, validation, and test datasets"""
        if use_filtered:
            # Use filtered data (no routine filings)
            data_path = '../../llm_features/filtered_data/'
            train_df = pd.read_csv(f'{data_path}filtered_train.csv')
            val_df = pd.read_csv(f'{data_path}filtered_val.csv')
            test_df = pd.read_csv(f'{data_path}filtered_test.csv')
        else:
            # Use original data
            data_path = '../data/'
            train_df = pd.read_csv(f'{data_path}train.csv')
            val_df = pd.read_csv(f'{data_path}val.csv')
            test_df = pd.read_csv(f'{data_path}test.csv')
        
        print(f"Data loaded from {data_path}:")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def prepare_targets(self, df, volatility_threshold=75):
        """Prepare different regression and classification targets"""
        targets = {}
        
        # REGRESSION TARGETS
        # 1. Raw returns (directional)
        targets['returns'] = df['adjusted_return_pct'].values
        
        # 2. Absolute returns (volatility)
        targets['volatility'] = np.abs(df['adjusted_return_pct'].values)
        
        # 3. Log volatility (for better distribution)
        targets['log_volatility'] = np.log1p(np.abs(df['adjusted_return_pct'].values))
        
        # 4. Squared returns (variance)
        targets['squared_returns'] = df['adjusted_return_pct'].values ** 2
        
        # CLASSIFICATION TARGETS
        # 5. High volatility binary (top 25% by default)
        threshold = np.percentile(np.abs(df['adjusted_return_pct'].values), volatility_threshold)
        targets['high_vol_binary'] = (np.abs(df['adjusted_return_pct'].values) > threshold).astype(int)
        
        # 6. Volatility tertiles (low, medium, high)
        vol = np.abs(df['adjusted_return_pct'].values)
        p33 = np.percentile(vol, 33)
        p67 = np.percentile(vol, 67)
        targets['vol_tertiles'] = np.digitize(vol, [p33, p67])
        
        return targets
    
    def display_target_statistics(self, train_targets, val_targets, test_targets):
        """Display statistics for different target variables"""
        print("\n" + "="*60)
        print("TARGET VARIABLE STATISTICS")
        print("="*60)
        
        for target_name in train_targets.keys():
            print(f"\n{target_name.upper()}:")
            print("-"*40)
            
            for split_name, targets in [('Train', train_targets), 
                                        ('Val', val_targets), 
                                        ('Test', test_targets)]:
                values = targets[target_name]
                print(f"  {split_name:6s} - Mean: {np.mean(values):6.3f}, "
                      f"Std: {np.std(values):6.3f}, "
                      f"Min: {np.min(values):6.3f}, "
                      f"Max: {np.max(values):6.3f}")
    
    def extract_llm_features(self, df, is_training=False):
        """Extract LLM-generated features from dataframe"""
        
        # Define feature groups (based on actual data types from analysis)
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
        
        # Market context features
        market_features = [
            'momentum_7d', 'momentum_30d', 'momentum_90d', 
            'momentum_365d', 'vix_level'
        ]
        
        # Initialize feature list
        features = []
        feature_names = []
        
        # Process numeric features
        numeric_data = []
        for feat in numeric_features:
            if feat in df.columns:
                numeric_data.append(df[feat].fillna(0).values.reshape(-1, 1))
                feature_names.append(f'llm_{feat}')
        
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
                        self.label_encoders[feat].fit(df[feat].fillna('unknown').astype(str))
                
                # Transform categorical to numeric
                if feat in self.label_encoders:
                    encoded = []
                    for val in df[feat].fillna('unknown').astype(str):
                        if val in self.label_encoders[feat].classes_:
                            encoded.append(self.label_encoders[feat].transform([val])[0])
                        else:
                            encoded.append(0)  # Default for unseen categories
                    features.append(np.array(encoded).reshape(-1, 1))
                    feature_names.append(f'llm_{feat}_encoded')
        
        # Process boolean features
        for feat in boolean_features:
            if feat in df.columns:
                features.append(df[feat].fillna(False).astype(int).values.reshape(-1, 1))
                feature_names.append(f'llm_{feat}')
        
        # Process market features
        market_data = []
        for feat in market_features:
            if feat in df.columns:
                market_data.append(df[feat].fillna(0).values.reshape(-1, 1))
                feature_names.append(f'market_{feat}')
        
        if market_data:
            market_array = np.hstack(market_data)
            features.append(market_array)
        
        # Store feature names for later analysis
        if is_training:
            self.llm_feature_names = feature_names
        
        # Combine all features
        if features:
            combined_features = np.hstack(features)
            return combined_features
        else:
            return np.zeros((len(df), 1))
    
    def create_features(self, train_df, val_df, test_df):
        """Create combined TF-IDF and LLM features"""
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION")
        print("="*60)
        
        # Extract text for TF-IDF
        print("\n1. Creating TF-IDF features...")
        train_text = train_df['summary'].fillna('').values
        val_text = val_df['summary'].fillna('').values
        test_text = test_df['summary'].fillna('').values
        
        # Create TF-IDF features with conservative parameters
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Reduced for regression
            ngram_range=(1, 2),
            min_df=10,
            max_df=0.7,
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
        self.feature_scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse compatibility
        X_train_llm_scaled = self.feature_scaler.fit_transform(X_train_llm)
        X_val_llm_scaled = self.feature_scaler.transform(X_val_llm)
        X_test_llm_scaled = self.feature_scaler.transform(X_test_llm)
        
        # Combine features
        print("\n4. Combining TF-IDF and LLM features...")
        X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_llm_scaled)])
        X_val_combined = hstack([X_val_tfidf, csr_matrix(X_val_llm_scaled)])
        X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_llm_scaled)])
        
        print(f"  Combined features shape: {X_train_combined.shape}")
        print(f"  Total features: {X_train_combined.shape[1]:,}")
        print(f"    - TF-IDF features: {X_train_tfidf.shape[1]:,}")
        print(f"    - LLM features: {X_train_llm.shape[1]:,}")
        
        # Also create TF-IDF only features for comparison
        features = {
            'combined': (X_train_combined, X_val_combined, X_test_combined),
            'tfidf_only': (X_train_tfidf, X_val_tfidf, X_test_tfidf),
            'llm_only': (csr_matrix(X_train_llm_scaled), 
                        csr_matrix(X_val_llm_scaled), 
                        csr_matrix(X_test_llm_scaled))
        }
        
        return features
    
    def train_model(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                   model_type='ridge', target_name='volatility'):
        """Train a regression model and evaluate performance"""
        
        # Select model
        if model_type == 'ridge':
            model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, random_state=42, max_iter=2000)
        elif model_type == 'elastic':
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000)
        elif model_type == 'rf':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'mlp':
            model = MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=128,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Ensure non-negative predictions for volatility
        if 'volatility' in target_name:
            train_pred = np.maximum(train_pred, 0)
            val_pred = np.maximum(val_pred, 0)
            test_pred = np.maximum(test_pred, 0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            y_train, train_pred,
            y_val, val_pred,
            y_test, test_pred
        )
        
        metrics['training_time'] = training_time
        metrics['model_type'] = model_type
        metrics['target'] = target_name
        
        return model, metrics
    
    def train_classifier(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                        model_type='logreg', target_name='high_vol_binary'):
        """Train a classification model for volatility event prediction"""
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        # Select model
        if model_type == 'logreg':
            model = LogisticRegression(
                C=0.1,
                penalty='l2',
                max_iter=1000,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=128,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # Get probabilities
        train_proba = model.predict_proba(X_train)[:, 1]
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train': {
                'accuracy': accuracy_score(y_train, train_pred),
                'precision': precision_score(y_train, train_pred),
                'recall': recall_score(y_train, train_pred),
                'f1': f1_score(y_train, train_pred),
                'auc': roc_auc_score(y_train, train_proba)
            },
            'val': {
                'accuracy': accuracy_score(y_val, val_pred),
                'precision': precision_score(y_val, val_pred),
                'recall': recall_score(y_val, val_pred),
                'f1': f1_score(y_val, val_pred),
                'auc': roc_auc_score(y_val, val_proba)
            },
            'test': {
                'accuracy': accuracy_score(y_test, test_pred),
                'precision': precision_score(y_test, test_pred),
                'recall': recall_score(y_test, test_pred),
                'f1': f1_score(y_test, test_pred),
                'auc': roc_auc_score(y_test, test_proba),
                'confusion_matrix': confusion_matrix(y_test, test_pred),
                'predictions': test_pred
            }
        }
        
        metrics['training_time'] = training_time
        metrics['model_type'] = model_type
        metrics['target'] = target_name
        
        return model, metrics
    
    def calculate_metrics(self, y_train, train_pred, y_val, val_pred, y_test, test_pred):
        """Calculate comprehensive regression metrics"""
        
        # Pearson correlation (primary metric)
        train_corr, train_p = pearsonr(y_train, train_pred)
        val_corr, val_p = pearsonr(y_val, val_pred)
        test_corr, test_p = pearsonr(y_test, test_pred)
        
        # Spearman correlation (rank correlation)
        train_spearman, _ = spearmanr(y_train, train_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        test_spearman, _ = spearmanr(y_test, test_pred)
        
        # R² score
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Mean Absolute Error
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Root Mean Squared Error
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        metrics = {
            'train': {
                'correlation': train_corr,
                'p_value': train_p,
                'spearman': train_spearman,
                'r2': train_r2,
                'mae': train_mae,
                'rmse': train_rmse
            },
            'val': {
                'correlation': val_corr,
                'p_value': val_p,
                'spearman': val_spearman,
                'r2': val_r2,
                'mae': val_mae,
                'rmse': val_rmse
            },
            'test': {
                'correlation': test_corr,
                'p_value': test_p,
                'spearman': test_spearman,
                'r2': test_r2,
                'mae': test_mae,
                'rmse': test_rmse,
                'predictions': test_pred
            }
        }
        
        return metrics
    
    def run_experiments(self, train_df, val_df, test_df):
        """Run comprehensive experiments with different models and targets"""
        
        # Prepare targets
        print("\nPreparing target variables...")
        train_targets = self.prepare_targets(train_df)
        val_targets = self.prepare_targets(val_df)
        test_targets = self.prepare_targets(test_df)
        
        # Display target statistics
        self.display_target_statistics(train_targets, val_targets, test_targets)
        
        # Create features
        features = self.create_features(train_df, val_df, test_df)
        
        # Define experiments
        regression_experiments = [
            # Volatility prediction (primary task)
            ('volatility', 'combined', 'ridge'),
            ('volatility', 'combined', 'elastic'),
            ('volatility', 'combined', 'mlp'),
            ('volatility', 'combined', 'gbm'),
            ('volatility', 'tfidf_only', 'ridge'),
            ('volatility', 'llm_only', 'ridge'),
            
            # Raw returns prediction (for comparison)
            ('returns', 'combined', 'ridge'),
            ('returns', 'combined', 'mlp'),
            ('returns', 'tfidf_only', 'ridge'),
            
            # Log volatility (better distribution)
            ('log_volatility', 'combined', 'ridge'),
            ('log_volatility', 'combined', 'mlp'),
        ]
        
        classification_experiments = [
            # High volatility event classification (top 25%)
            ('high_vol_binary', 'combined', 'logreg'),
            ('high_vol_binary', 'combined', 'rf'),
            ('high_vol_binary', 'combined', 'mlp'),
            ('high_vol_binary', 'tfidf_only', 'logreg'),
            ('high_vol_binary', 'llm_only', 'logreg'),
        ]
        
        all_results = []
        
        print("\n" + "="*60)
        print("RUNNING REGRESSION EXPERIMENTS")
        print("="*60)
        
        for target_name, feature_set, model_type in regression_experiments:
            print(f"\n{target_name.upper()} - {feature_set} - {model_type}")
            print("-"*40)
            
            # Get features and targets
            X_train, X_val, X_test = features[feature_set]
            y_train = train_targets[target_name]
            y_val = val_targets[target_name]
            y_test = test_targets[target_name]
            
            # Train model
            model, metrics = self.train_model(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                model_type=model_type,
                target_name=target_name
            )
            
            # Store results
            result = {
                'target': target_name,
                'features': feature_set,
                'model': model_type,
                'test_correlation': metrics['test']['correlation'],
                'test_r2': metrics['test']['r2'],
                'test_mae': metrics['test']['mae'],
                'val_correlation': metrics['val']['correlation'],
                'train_test_corr_gap': metrics['train']['correlation'] - metrics['test']['correlation']
            }
            
            all_results.append(result)
            
            # Display key metrics
            print(f"  Test Correlation: {metrics['test']['correlation']:.4f} (p={metrics['test']['p_value']:.2e})")
            print(f"  Test R²: {metrics['test']['r2']:.4f}")
            print(f"  Test MAE: {metrics['test']['mae']:.3f}")
            print(f"  Overfitting (train-test corr gap): {result['train_test_corr_gap']:.4f}")
        
        # Run classification experiments
        print("\n" + "="*60)
        print("RUNNING CLASSIFICATION EXPERIMENTS")
        print("(Predicting High Volatility Events - Top 25%)")
        print("="*60)
        
        # Display class distribution
        high_vol_train = train_targets['high_vol_binary']
        high_vol_test = test_targets['high_vol_binary']
        print(f"\nClass Distribution:")
        print(f"  Train - High vol: {high_vol_train.mean():.1%}, Normal: {1-high_vol_train.mean():.1%}")
        print(f"  Test  - High vol: {high_vol_test.mean():.1%}, Normal: {1-high_vol_test.mean():.1%}")
        
        classification_results = []
        
        for target_name, feature_set, model_type in classification_experiments:
            print(f"\n{target_name.upper()} - {feature_set} - {model_type}")
            print("-"*40)
            
            # Get features and targets
            X_train, X_val, X_test = features[feature_set]
            y_train = train_targets[target_name]
            y_val = val_targets[target_name]
            y_test = test_targets[target_name]
            
            # Train classifier
            model, metrics = self.train_classifier(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                model_type=model_type,
                target_name=target_name
            )
            
            # Store results
            result = {
                'task': 'classification',
                'target': target_name,
                'features': feature_set,
                'model': model_type,
                'test_accuracy': metrics['test']['accuracy'],
                'test_precision': metrics['test']['precision'],
                'test_recall': metrics['test']['recall'],
                'test_f1': metrics['test']['f1'],
                'test_auc': metrics['test']['auc'],
                'val_f1': metrics['val']['f1'],
                'train_test_acc_gap': metrics['train']['accuracy'] - metrics['test']['accuracy']
            }
            
            classification_results.append(result)
            all_results.append(result)
            
            # Display key metrics
            print(f"  Test Accuracy:  {metrics['test']['accuracy']:.1%}")
            print(f"  Test Precision: {metrics['test']['precision']:.1%} (when predicting high vol)")
            print(f"  Test Recall:    {metrics['test']['recall']:.1%} (% of high vol caught)")
            print(f"  Test F1 Score:  {metrics['test']['f1']:.3f}")
            print(f"  Test AUC:       {metrics['test']['auc']:.3f}")
            
            # Display confusion matrix
            cm = metrics['test']['confusion_matrix']
            print(f"\n  Confusion Matrix:")
            print(f"              Pred Normal  Pred High")
            print(f"  Act Normal:    {cm[0,0]:5d}      {cm[0,1]:5d}")
            print(f"  Act High:      {cm[1,0]:5d}      {cm[1,1]:5d}")
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    def analyze_results(self):
        """Analyze and display experiment results"""
        
        print("\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        # Separate regression and classification results
        regression_results = self.results[self.results.get('task', '') != 'classification']
        classification_results = self.results[self.results.get('task', '') == 'classification']
        
        # Best models for volatility regression
        print("\n1. VOLATILITY REGRESSION (Predicting Magnitude)")
        print("-"*40)
        volatility_results = regression_results[regression_results['target'] == 'volatility'].sort_values(
            'test_correlation', ascending=False
        )
        
        print("\nTop 3 models by correlation:")
        for idx, row in volatility_results.head(3).iterrows():
            print(f"  {row['features']:15s} + {row['model']:8s}: "
                  f"Corr={row['test_correlation']:.4f}, R²={row['test_r2']:.4f}")
        
        # Best models for volatility classification
        if not classification_results.empty:
            print("\n2. VOLATILITY CLASSIFICATION (High vs Normal Events)")
            print("-"*40)
            vol_class_results = classification_results[
                classification_results['target'] == 'high_vol_binary'
            ].sort_values('test_f1', ascending=False)
            
            print("\nTop models by F1 score:")
            for idx, row in vol_class_results.head(3).iterrows():
                print(f"  {row['features']:15s} + {row['model']:8s}: "
                      f"F1={row['test_f1']:.3f}, Precision={row['test_precision']:.1%}, "
                      f"Recall={row['test_recall']:.1%}")
        
        # Feature set comparison for volatility
        print("\n3. FEATURE SET COMPARISON (Volatility Regression, Ridge)")
        print("-"*40)
        feature_comparison = regression_results[
            (regression_results['target'] == 'volatility') & 
            (regression_results['model'] == 'ridge')
        ].sort_values('test_correlation', ascending=False)
        
        for idx, row in feature_comparison.iterrows():
            print(f"  {row['features']:15s}: Corr={row['test_correlation']:.4f}, "
                  f"R²={row['test_r2']:.4f}, MAE={row['test_mae']:.3f}")
        
        # Returns vs Volatility
        print("\n3. RETURNS vs VOLATILITY PREDICTION")
        print("-"*40)
        
        returns_best = self.results[
            (self.results['target'] == 'returns') & 
            (self.results['features'] == 'combined')
        ].sort_values('test_correlation', ascending=False).iloc[0]
        
        volatility_best = self.results[
            (self.results['target'] == 'volatility') & 
            (self.results['features'] == 'combined')
        ].sort_values('test_correlation', ascending=False).iloc[0]
        
        print(f"  Best Returns Prediction:    Corr={returns_best['test_correlation']:.4f}, "
              f"R²={returns_best['test_r2']:.4f}")
        print(f"  Best Volatility Prediction: Corr={volatility_best['test_correlation']:.4f}, "
              f"R²={volatility_best['test_r2']:.4f}")
        
        improvement = (volatility_best['test_correlation'] - abs(returns_best['test_correlation'])) / abs(returns_best['test_correlation']) * 100
        print(f"  Volatility prediction {improvement:.1f}% better than returns prediction")
        
        # Log volatility comparison
        print("\n4. LOG TRANSFORMATION EFFECT")
        print("-"*40)
        
        log_vol = self.results[
            (self.results['target'] == 'log_volatility') & 
            (self.results['features'] == 'combined') &
            (self.results['model'] == 'ridge')
        ]
        
        regular_vol = self.results[
            (self.results['target'] == 'volatility') & 
            (self.results['features'] == 'combined') &
            (self.results['model'] == 'ridge')
        ]
        
        if not log_vol.empty and not regular_vol.empty:
            print(f"  Regular Volatility: Corr={regular_vol.iloc[0]['test_correlation']:.4f}")
            print(f"  Log Volatility:     Corr={log_vol.iloc[0]['test_correlation']:.4f}")
        
        # Save results
        self.results.to_csv('regression_results.csv', index=False)
        print("\n\nDetailed results saved to: regression_results.csv")
    
    def feature_importance_analysis(self, train_df, val_df, test_df):
        """Analyze feature importance for best model"""
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Use Ridge regression for interpretability
        features = self.create_features(train_df, val_df, test_df)
        X_train, _, X_test = features['combined']
        
        train_targets = self.prepare_targets(train_df)
        test_targets = self.prepare_targets(test_df)
        
        y_train = train_targets['volatility']
        y_test = test_targets['volatility']
        
        # Train Ridge model
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Get coefficients
        coefficients = model.coef_
        
        # Split by feature type
        tfidf_size = len(self.vectorizer.get_feature_names_out())
        tfidf_coefs = coefficients[:tfidf_size]
        llm_coefs = coefficients[tfidf_size:]
        
        # Top TF-IDF features for volatility
        print("\nTop 15 TF-IDF features for HIGH volatility:")
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': tfidf_coefs
        }).sort_values('coefficient', ascending=False)
        
        for idx, row in tfidf_importance.head(15).iterrows():
            print(f"  {row['feature']:30s} {row['coefficient']:+.4f}")
        
        print("\nTop 15 TF-IDF features for LOW volatility:")
        for idx, row in tfidf_importance.tail(15).iterrows():
            print(f"  {row['feature']:30s} {row['coefficient']:+.4f}")
        
        # LLM features analysis
        print("\n\nLLM Features Impact Summary:")
        print("-"*40)
        print(f"  Number of TF-IDF features: {len(tfidf_coefs)}")
        print(f"  Number of LLM features: {len(llm_coefs)}")
        print(f"  Average |coefficient| (TF-IDF): {np.abs(tfidf_coefs).mean():.4f}")
        print(f"  Average |coefficient| (LLM):    {np.abs(llm_coefs).mean():.4f}")
        print(f"  Max |coefficient| (TF-IDF):     {np.abs(tfidf_coefs).max():.4f}")
        print(f"  Max |coefficient| (LLM):        {np.abs(llm_coefs).max():.4f}")
        
        # Top LLM features
        if hasattr(self, 'llm_feature_names') and len(self.llm_feature_names) == len(llm_coefs):
            print("\nTop 10 LLM features by importance:")
            llm_importance = pd.DataFrame({
                'feature': self.llm_feature_names,
                'coefficient': np.abs(llm_coefs)
            }).sort_values('coefficient', ascending=False)
            
            for idx, row in llm_importance.head(10).iterrows():
                print(f"  {row['feature']:30s} {row['coefficient']:.4f}")

def main():
    """Main execution function"""
    
    print("="*60)
    print("COMBINED FEATURES REGRESSION ANALYSIS")
    print("Predicting Volatility and Returns with TF-IDF + LLM Features")
    print("="*60)
    
    # Initialize regressor
    regressor = CombinedVolatilityRegressor()
    
    # Load data (using filtered data by default)
    train_df, val_df, test_df = regressor.load_data(use_filtered=True)
    
    # Run experiments
    results_df = regressor.run_experiments(train_df, val_df, test_df)
    
    # Analyze results
    regressor.analyze_results()
    
    # Feature importance analysis
    regressor.feature_importance_analysis(train_df, val_df, test_df)
    
    # Final summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    best_result = results_df.sort_values('test_correlation', ascending=False).iloc[0]
    
    print(f"\n1. Best Model Configuration:")
    print(f"   Target: {best_result['target']}")
    print(f"   Features: {best_result['features']}")
    print(f"   Model: {best_result['model']}")
    print(f"   Test Correlation: {best_result['test_correlation']:.4f}")
    print(f"   Test R²: {best_result['test_r2']:.4f}")
    
    # Compare feature sets for volatility
    vol_combined = results_df[
        (results_df['target'] == 'volatility') & 
        (results_df['features'] == 'combined') &
        (results_df['model'] == 'ridge')
    ].iloc[0]['test_correlation']
    
    vol_tfidf = results_df[
        (results_df['target'] == 'volatility') & 
        (results_df['features'] == 'tfidf_only') &
        (results_df['model'] == 'ridge')
    ].iloc[0]['test_correlation']
    
    vol_llm = results_df[
        (results_df['target'] == 'volatility') & 
        (results_df['features'] == 'llm_only') &
        (results_df['model'] == 'ridge')
    ].iloc[0]['test_correlation']
    
    print(f"\n2. Feature Set Contribution (Volatility, Ridge):")
    print(f"   TF-IDF only:     {vol_tfidf:.4f}")
    print(f"   LLM only:        {vol_llm:.4f}")
    print(f"   Combined:        {vol_combined:.4f}")
    
    if vol_combined > vol_tfidf:
        improvement = (vol_combined - vol_tfidf) / vol_tfidf * 100
        print(f"   → LLM features improve correlation by {improvement:.1f}%")
    else:
        print(f"   → LLM features do not improve performance")
    
    # Baseline comparisons
    print(f"\n3. Baseline Comparisons:")
    print(f"   Random predictions correlation: ~0.00")
    print(f"   Our best correlation: {best_result['test_correlation']:.4f}")
    
    if best_result['test_correlation'] > 0.15:
        print(f"   ✓ Model shows meaningful predictive signal")
    elif best_result['test_correlation'] > 0.10:
        print(f"   ⚠ Model shows weak predictive signal")
    else:
        print(f"   ✗ Model shows very weak predictive signal")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Save final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_model': {
            'target': best_result['target'],
            'features': best_result['features'],
            'model': best_result['model'],
            'test_correlation': float(best_result['test_correlation']),
            'test_r2': float(best_result['test_r2']),
            'test_mae': float(best_result['test_mae'])
        },
        'feature_comparison': {
            'tfidf_only': float(vol_tfidf),
            'llm_only': float(vol_llm),
            'combined': float(vol_combined)
        }
    }
    
    with open('results/regression_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary saved to: regression_summary.json")

if __name__ == "__main__":
    main()