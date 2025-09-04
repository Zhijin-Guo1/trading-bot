#!/usr/bin/env python3
"""
Feature Set Comparison Analysis
================================
Compares three feature configurations:
1. TF-IDF only
2. TF-IDF + LLM features (no market context)
3. TF-IDF + LLM features + Market Context (full features)

Generates comparison plots and feature importance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class FeatureSetComparator:
    """Compare different feature set configurations"""
    
    def __init__(self):
        self.vectorizer = None
        self.feature_scaler = None
        self.label_encoders = {}
        self.results = []
        self.models = {}
        self.feature_importances = {}
        
    def load_data(self):
        """Load train, validation, and test datasets"""
        data_path = '../../llm_features/filtered_data/'
        
        train_df = pd.read_csv(f'{data_path}filtered_train.csv')
        val_df = pd.read_csv(f'{data_path}filtered_val.csv')
        test_df = pd.read_csv(f'{data_path}filtered_test.csv')
        
        print(f"Data loaded:")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def prepare_targets(self, df):
        """Prepare volatility target"""
        return np.abs(df['adjusted_return_pct'].values)
    
    def extract_tfidf_features(self, train_text, val_text, test_text):
        """Extract TF-IDF features"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=10,
            max_df=0.7,
            stop_words='english',
            sublinear_tf=True,
            norm='l2'
        )
        
        X_train = self.vectorizer.fit_transform(train_text)
        X_val = self.vectorizer.transform(val_text)
        X_test = self.vectorizer.transform(test_text)
        
        return X_train, X_val, X_test
    
    def extract_llm_features(self, df, is_training=False, include_market=False):
        """Extract LLM features with optional market context"""
        
        # Core LLM features (linguistic only)
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
        
        # Market context features (optional)
        market_features = []
        if include_market:
            market_features = [
                'momentum_7d', 'momentum_30d', 'momentum_90d',
                'momentum_365d', 'vix_level'
            ]
        
        features = []
        feature_names = []
        
        # Process numeric features
        for feat in numeric_features:
            if feat in df.columns:
                features.append(df[feat].fillna(0).values.reshape(-1, 1))
                feature_names.append(f'llm_{feat}')
        
        # Process categorical features
        for feat in categorical_features:
            if feat in df.columns:
                if is_training:
                    if feat not in self.label_encoders:
                        self.label_encoders[feat] = LabelEncoder()
                        self.label_encoders[feat].fit(df[feat].fillna('unknown').astype(str))
                
                if feat in self.label_encoders:
                    encoded = []
                    for val in df[feat].fillna('unknown').astype(str):
                        if val in self.label_encoders[feat].classes_:
                            encoded.append(self.label_encoders[feat].transform([val])[0])
                        else:
                            encoded.append(0)
                    features.append(np.array(encoded).reshape(-1, 1))
                    feature_names.append(f'llm_{feat}_encoded')
        
        # Process boolean features
        for feat in boolean_features:
            if feat in df.columns:
                features.append(df[feat].fillna(False).astype(int).values.reshape(-1, 1))
                feature_names.append(f'llm_{feat}')
        
        # Process market features if included
        if include_market:
            for feat in market_features:
                if feat in df.columns:
                    features.append(df[feat].fillna(0).values.reshape(-1, 1))
                    feature_names.append(f'market_{feat}')
        
        if is_training:
            if include_market:
                self.llm_market_feature_names = feature_names
            else:
                self.llm_only_feature_names = feature_names
        
        if features:
            return np.hstack(features), feature_names
        else:
            return np.zeros((len(df), 1)), []
    
    def create_feature_sets(self, train_df, val_df, test_df):
        """Create three different feature set configurations"""
        
        print("\n" + "="*60)
        print("CREATING FEATURE SETS")
        print("="*60)
        
        # Extract text for TF-IDF
        train_text = train_df['summary'].fillna('').values
        val_text = val_df['summary'].fillna('').values
        test_text = test_df['summary'].fillna('').values
        
        # 1. TF-IDF Only
        print("\n1. TF-IDF Only Features...")
        X_train_tfidf, X_val_tfidf, X_test_tfidf = self.extract_tfidf_features(
            train_text, val_text, test_text
        )
        print(f"   Shape: {X_train_tfidf.shape}")
        
        # 2. TF-IDF + LLM (no market context)
        print("\n2. TF-IDF + LLM Features (no market context)...")
        X_train_llm, llm_names = self.extract_llm_features(train_df, is_training=True, include_market=False)
        X_val_llm, _ = self.extract_llm_features(val_df, is_training=False, include_market=False)
        X_test_llm, _ = self.extract_llm_features(test_df, is_training=False, include_market=False)
        
        # Scale LLM features
        self.feature_scaler = StandardScaler(with_mean=False)
        X_train_llm_scaled = self.feature_scaler.fit_transform(X_train_llm)
        X_val_llm_scaled = self.feature_scaler.transform(X_val_llm)
        X_test_llm_scaled = self.feature_scaler.transform(X_test_llm)
        
        X_train_tfidf_llm = hstack([X_train_tfidf, csr_matrix(X_train_llm_scaled)])
        X_val_tfidf_llm = hstack([X_val_tfidf, csr_matrix(X_val_llm_scaled)])
        X_test_tfidf_llm = hstack([X_test_tfidf, csr_matrix(X_test_llm_scaled)])
        print(f"   Shape: {X_train_tfidf_llm.shape}")
        print(f"   (TF-IDF: {X_train_tfidf.shape[1]}, LLM: {X_train_llm.shape[1]})")
        
        # 3. TF-IDF + LLM + Market Context (full)
        print("\n3. TF-IDF + LLM + Market Context (full features)...")
        X_train_llm_market, llm_market_names = self.extract_llm_features(
            train_df, is_training=True, include_market=True
        )
        X_val_llm_market, _ = self.extract_llm_features(val_df, is_training=False, include_market=True)
        X_test_llm_market, _ = self.extract_llm_features(test_df, is_training=False, include_market=True)
        
        # Scale full LLM+market features
        self.full_feature_scaler = StandardScaler(with_mean=False)
        X_train_llm_market_scaled = self.full_feature_scaler.fit_transform(X_train_llm_market)
        X_val_llm_market_scaled = self.full_feature_scaler.transform(X_val_llm_market)
        X_test_llm_market_scaled = self.full_feature_scaler.transform(X_test_llm_market)
        
        X_train_full = hstack([X_train_tfidf, csr_matrix(X_train_llm_market_scaled)])
        X_val_full = hstack([X_val_tfidf, csr_matrix(X_val_llm_market_scaled)])
        X_test_full = hstack([X_test_tfidf, csr_matrix(X_test_llm_market_scaled)])
        print(f"   Shape: {X_train_full.shape}")
        print(f"   (TF-IDF: {X_train_tfidf.shape[1]}, LLM+Market: {X_train_llm_market.shape[1]})")
        
        feature_sets = {
            'TF-IDF Only': {
                'train': X_train_tfidf,
                'val': X_val_tfidf,
                'test': X_test_tfidf,
                'n_features': X_train_tfidf.shape[1],
                'feature_names': list(self.vectorizer.get_feature_names_out())
            },
            'TF-IDF + LLM': {
                'train': X_train_tfidf_llm,
                'val': X_val_tfidf_llm,
                'test': X_test_tfidf_llm,
                'n_features': X_train_tfidf_llm.shape[1],
                'feature_names': list(self.vectorizer.get_feature_names_out()) + llm_names
            },
            'TF-IDF + LLM + Market': {
                'train': X_train_full,
                'val': X_val_full,
                'test': X_test_full,
                'n_features': X_train_full.shape[1],
                'feature_names': list(self.vectorizer.get_feature_names_out()) + llm_market_names
            }
        }
        
        return feature_sets
    
    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                           model_type='ridge', feature_set_name=''):
        """Train model and evaluate performance"""
        
        # Select model
        if model_type == 'ridge':
            model = Ridge(alpha=1.0, random_state=42)
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                learning_rate=0.1,
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
        
        # Ensure non-negative predictions
        train_pred = np.maximum(train_pred, 0)
        val_pred = np.maximum(val_pred, 0)
        test_pred = np.maximum(test_pred, 0)
        
        # Calculate metrics
        train_pearson, _ = pearsonr(y_train, train_pred)
        val_pearson, _ = pearsonr(y_val, val_pred)
        test_pearson, _ = pearsonr(y_test, test_pred)
        
        train_spearman, _ = spearmanr(y_train, train_pred)
        val_spearman, _ = spearmanr(y_val, val_pred)
        test_spearman, _ = spearmanr(y_test, test_pred)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        metrics = {
            'feature_set': feature_set_name,
            'model_type': model_type,
            'train_pearson': train_pearson,
            'val_pearson': val_pearson,
            'test_pearson': test_pearson,
            'train_spearman': train_spearman,
            'val_spearman': val_spearman,
            'test_spearman': test_spearman,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'training_time': training_time,
            'overfitting_gap': train_pearson - test_pearson
        }
        
        return model, metrics
    
    def run_comparison(self):
        """Run full comparison across feature sets"""
        
        print("\n" + "="*60)
        print("RUNNING FEATURE SET COMPARISON")
        print("="*60)
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Prepare targets
        y_train = self.prepare_targets(train_df)
        y_val = self.prepare_targets(val_df)
        y_test = self.prepare_targets(test_df)
        
        # Create feature sets
        feature_sets = self.create_feature_sets(train_df, val_df, test_df)
        
        # Run experiments for each feature set
        model_types = ['ridge', 'gbm']
        
        for feature_set_name, feature_data in feature_sets.items():
            print(f"\n\nEvaluating: {feature_set_name}")
            print("-" * 40)
            
            X_train = feature_data['train']
            X_val = feature_data['val']
            X_test = feature_data['test']
            
            for model_type in model_types:
                print(f"\n  Model: {model_type}")
                
                model, metrics = self.train_and_evaluate(
                    X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    model_type=model_type,
                    feature_set_name=feature_set_name
                )
                
                # Store model for feature importance
                self.models[f"{feature_set_name}_{model_type}"] = model
                
                # Store results
                self.results.append(metrics)
                
                # Display key metrics
                print(f"    Test Pearson:  {metrics['test_pearson']:.4f}")
                print(f"    Test Spearman: {metrics['test_spearman']:.4f}")
                print(f"    Test R²:       {metrics['test_r2']:.4f}")
                print(f"    Test MAE:      {metrics['test_mae']:.3f}")
                print(f"    Overfitting:   {metrics['overfitting_gap']:.4f}")
                
                # Extract feature importance for Ridge models
                if model_type == 'ridge':
                    coefficients = model.coef_
                    feature_names = feature_data['feature_names']
                    
                    # Store for later analysis
                    self.feature_importances[feature_set_name] = {
                        'coefficients': coefficients,
                        'feature_names': feature_names
                    }
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        return self.results_df
    
    def create_comparison_plots(self):
        """Create comprehensive comparison plots"""
        
        print("\n" + "="*60)
        print("CREATING COMPARISON PLOTS")
        print("="*60)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Feature Set Comparison: TF-IDF vs TF-IDF+LLM vs Full Features', 
                     fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        feature_sets = ['TF-IDF Only', 'TF-IDF + LLM', 'TF-IDF + LLM + Market']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        # 1. Pearson Correlation Comparison
        ax = axes[0, 0]
        for model_type in ['ridge', 'gbm']:
            data = self.results_df[self.results_df['model_type'] == model_type]
            x = np.arange(len(feature_sets))
            width = 0.35
            offset = -width/2 if model_type == 'ridge' else width/2
            
            values = [data[data['feature_set'] == fs]['test_pearson'].values[0] 
                     for fs in feature_sets]
            
            ax.bar(x + offset, values, width, 
                  label=model_type.upper(), alpha=0.8)
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Test Set Pearson Correlation')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Spearman Correlation Comparison
        ax = axes[0, 1]
        for model_type in ['ridge', 'gbm']:
            data = self.results_df[self.results_df['model_type'] == model_type]
            x = np.arange(len(feature_sets))
            width = 0.35
            offset = -width/2 if model_type == 'ridge' else width/2
            
            values = [data[data['feature_set'] == fs]['test_spearman'].values[0] 
                     for fs in feature_sets]
            
            ax.bar(x + offset, values, width, 
                  label=model_type.upper(), alpha=0.8)
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Test Set Spearman Correlation')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. R² Score Comparison
        ax = axes[0, 2]
        for model_type in ['ridge', 'gbm']:
            data = self.results_df[self.results_df['model_type'] == model_type]
            x = np.arange(len(feature_sets))
            width = 0.35
            offset = -width/2 if model_type == 'ridge' else width/2
            
            values = [data[data['feature_set'] == fs]['test_r2'].values[0] 
                     for fs in feature_sets]
            
            ax.bar(x + offset, values, width, 
                  label=model_type.upper(), alpha=0.8)
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('R² Score')
        ax.set_title('Test Set R² Score')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. MAE Comparison
        ax = axes[1, 0]
        for model_type in ['ridge', 'gbm']:
            data = self.results_df[self.results_df['model_type'] == model_type]
            x = np.arange(len(feature_sets))
            width = 0.35
            offset = -width/2 if model_type == 'ridge' else width/2
            
            values = [data[data['feature_set'] == fs]['test_mae'].values[0] 
                     for fs in feature_sets]
            
            ax.bar(x + offset, values, width, 
                  label=model_type.upper(), alpha=0.8)
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Test Set MAE (Lower is Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Overfitting Analysis
        ax = axes[1, 1]
        for model_type in ['ridge', 'gbm']:
            data = self.results_df[self.results_df['model_type'] == model_type]
            x = np.arange(len(feature_sets))
            width = 0.35
            offset = -width/2 if model_type == 'ridge' else width/2
            
            values = [data[data['feature_set'] == fs]['overfitting_gap'].values[0] 
                     for fs in feature_sets]
            
            ax.bar(x + offset, values, width, 
                  label=model_type.upper(), alpha=0.8)
        
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Train-Test Gap')
        ax.set_title('Overfitting (Lower is Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # 6. Performance Improvement
        ax = axes[1, 2]
        ridge_data = self.results_df[self.results_df['model_type'] == 'ridge']
        baseline = ridge_data[ridge_data['feature_set'] == 'TF-IDF Only']['test_spearman'].values[0]
        
        improvements = []
        for fs in feature_sets:
            value = ridge_data[ridge_data['feature_set'] == fs]['test_spearman'].values[0]
            improvement = ((value - baseline) / baseline) * 100 if baseline != 0 else 0
            improvements.append(improvement)
        
        bars = ax.bar(feature_sets, improvements, color=colors, alpha=0.8)
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Spearman Correlation Improvement vs TF-IDF Only (Ridge)')
        ax.set_xticklabels(feature_sets, rotation=15, ha='right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('feature_set_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("Comparison plots saved to: feature_set_comparison.png")
    
    def create_feature_importance_plot(self):
        """Create feature importance visualization"""
        
        print("\n" + "="*60)
        print("CREATING FEATURE IMPORTANCE PLOT")
        print("="*60)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Feature Importance Analysis (Ridge Coefficients)', 
                     fontsize=16, fontweight='bold')
        
        for idx, (feature_set_name, data) in enumerate(self.feature_importances.items()):
            ax = axes[idx]
            
            coefficients = data['coefficients']
            feature_names = data['feature_names']
            
            # Separate TF-IDF and other features
            n_tfidf = 1000  # We know TF-IDF has 1000 features
            
            # Calculate aggregate importance by feature type
            tfidf_importance = np.abs(coefficients[:n_tfidf]).mean()
            
            if feature_set_name == 'TF-IDF Only':
                # Only TF-IDF features
                importance_dict = {
                    'TF-IDF (avg)': tfidf_importance,
                }
                
                # Add top 10 individual TF-IDF features
                tfidf_df = pd.DataFrame({
                    'feature': feature_names[:n_tfidf],
                    'importance': np.abs(coefficients[:n_tfidf])
                }).sort_values('importance', ascending=False)
                
                for _, row in tfidf_df.head(10).iterrows():
                    importance_dict[row['feature']] = row['importance']
                
            else:
                # TF-IDF + other features
                other_coeffs = coefficients[n_tfidf:]
                other_names = feature_names[n_tfidf:]
                
                # Aggregate by feature type
                llm_features = [i for i, name in enumerate(other_names) if name.startswith('llm_')]
                market_features = [i for i, name in enumerate(other_names) if name.startswith('market_')]
                
                importance_dict = {
                    'TF-IDF (avg)': tfidf_importance,
                }
                
                if llm_features:
                    llm_importance = np.abs(other_coeffs[llm_features]).mean()
                    importance_dict['LLM Features (avg)'] = llm_importance
                    
                    # Add top 5 LLM features
                    llm_df = pd.DataFrame({
                        'feature': [other_names[i] for i in llm_features],
                        'importance': np.abs(other_coeffs[llm_features])
                    }).sort_values('importance', ascending=False)
                    
                    for _, row in llm_df.head(5).iterrows():
                        clean_name = row['feature'].replace('llm_', '').replace('_', ' ').title()
                        importance_dict[clean_name] = row['importance']
                
                if market_features:
                    market_importance = np.abs(other_coeffs[market_features]).mean()
                    importance_dict['Market Context (avg)'] = market_importance
                    
                    # Add individual market features
                    for i in market_features:
                        clean_name = other_names[i].replace('market_', '').replace('_', ' ').title()
                        importance_dict[clean_name] = np.abs(other_coeffs[i])
            
            # Create bar plot
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            
            # Color code by type
            colors = []
            for feat in features:
                if '(avg)' in feat:
                    colors.append('#3498db')  # Blue for aggregates
                elif any(x in feat.lower() for x in ['momentum', 'vix']):
                    colors.append('#e74c3c')  # Red for market
                elif feat.startswith('TF-IDF') or len(feat.split()) <= 2:
                    colors.append('#95a5a6')  # Gray for TF-IDF words
                else:
                    colors.append('#2ecc71')  # Green for LLM features
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Absolute Coefficient Value')
            ax.set_title(f'{feature_set_name}')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (feat, val) in enumerate(zip(features, importances)):
                ax.annotate(f'{val:.4f}', 
                           xy=(val, i),
                           xytext=(3, 0),
                           textcoords='offset points',
                           va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('feature_importance_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("Feature importance plot saved to: feature_importance_comparison.png")
    
    def save_results(self):
        """Save results to CSV with enhanced metrics"""
        
        # Add percentage improvements
        ridge_baseline = self.results_df[
            (self.results_df['model_type'] == 'ridge') & 
            (self.results_df['feature_set'] == 'TF-IDF Only')
        ]['test_spearman'].values[0]
        
        self.results_df['spearman_improvement_pct'] = self.results_df.apply(
            lambda row: ((row['test_spearman'] - ridge_baseline) / ridge_baseline * 100) 
            if row['model_type'] == 'ridge' else np.nan,
            axis=1
        )
        
        # Save to CSV
        self.results_df.to_csv('feature_set_comparison_results.csv', index=False)
        print("\nResults saved to: feature_set_comparison_results.csv")
        
        # Save summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'feature_sets': {
                'tfidf_only': {
                    'n_features': 1000,
                    'best_spearman': self.results_df[
                        self.results_df['feature_set'] == 'TF-IDF Only'
                    ]['test_spearman'].max()
                },
                'tfidf_llm': {
                    'n_features': self.results_df[
                        self.results_df['feature_set'] == 'TF-IDF + LLM'
                    ].iloc[0]['feature_set'],  # Will update this
                    'best_spearman': self.results_df[
                        self.results_df['feature_set'] == 'TF-IDF + LLM'
                    ]['test_spearman'].max()
                },
                'full_features': {
                    'n_features': self.results_df[
                        self.results_df['feature_set'] == 'TF-IDF + LLM + Market'
                    ].iloc[0]['feature_set'],  # Will update this
                    'best_spearman': self.results_df[
                        self.results_df['feature_set'] == 'TF-IDF + LLM + Market'
                    ]['test_spearman'].max()
                }
            },
            'key_insights': {
                'llm_improvement': f"{((self.results_df[self.results_df['feature_set'] == 'TF-IDF + LLM']['test_spearman'].max() - ridge_baseline) / ridge_baseline * 100):.1f}%",
                'market_improvement': f"{((self.results_df[self.results_df['feature_set'] == 'TF-IDF + LLM + Market']['test_spearman'].max() - ridge_baseline) / ridge_baseline * 100):.1f}%"
            }
        }
        
        with open('feature_comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Summary saved to: feature_comparison_summary.json")
    
    def display_summary(self):
        """Display key findings summary"""
        
        print("\n" + "="*60)
        print("KEY FINDINGS SUMMARY")
        print("="*60)
        
        # Best performance by feature set
        print("\nBest Spearman Correlation by Feature Set:")
        print("-" * 40)
        for feature_set in ['TF-IDF Only', 'TF-IDF + LLM', 'TF-IDF + LLM + Market']:
            best = self.results_df[
                self.results_df['feature_set'] == feature_set
            ].sort_values('test_spearman', ascending=False).iloc[0]
            print(f"{feature_set:25s}: {best['test_spearman']:.4f} ({best['model_type']})")
        
        # Incremental improvements
        print("\nIncremental Improvements (Ridge model):")
        print("-" * 40)
        ridge_data = self.results_df[self.results_df['model_type'] == 'ridge']
        
        tfidf_only = ridge_data[ridge_data['feature_set'] == 'TF-IDF Only']['test_spearman'].values[0]
        tfidf_llm = ridge_data[ridge_data['feature_set'] == 'TF-IDF + LLM']['test_spearman'].values[0]
        full = ridge_data[ridge_data['feature_set'] == 'TF-IDF + LLM + Market']['test_spearman'].values[0]
        
        llm_boost = ((tfidf_llm - tfidf_only) / tfidf_only) * 100
        market_boost = ((full - tfidf_llm) / tfidf_llm) * 100
        total_boost = ((full - tfidf_only) / tfidf_only) * 100
        
        print(f"TF-IDF baseline:          {tfidf_only:.4f}")
        print(f"Adding LLM features:      +{llm_boost:.1f}% → {tfidf_llm:.4f}")
        print(f"Adding Market context:    +{market_boost:.1f}% → {full:.4f}")
        print(f"Total improvement:        +{total_boost:.1f}%")
        
        # Feature efficiency
        print("\nFeature Efficiency Analysis:")
        print("-" * 40)
        print(f"TF-IDF: 1000 features → {tfidf_only:.4f} correlation")
        print(f"  Efficiency: {tfidf_only/1000*1000:.2f} correlation per 1000 features")
        
        llm_n = 20  # Approximate number of LLM features
        llm_contribution = tfidf_llm - tfidf_only
        print(f"\nLLM: ~{llm_n} features → +{llm_contribution:.4f} correlation")
        print(f"  Efficiency: {llm_contribution/llm_n*1000:.2f} correlation per 1000 features")
        print(f"  LLM features are {llm_contribution/llm_n*1000/(tfidf_only/1000*1000):.1f}x more efficient")
        
        market_n = 5  # Number of market features
        market_contribution = full - tfidf_llm
        print(f"\nMarket: {market_n} features → +{market_contribution:.4f} correlation")
        print(f"  Efficiency: {market_contribution/market_n*1000:.2f} correlation per 1000 features")
        print(f"  Market features are {market_contribution/market_n*1000/(tfidf_only/1000*1000):.1f}x more efficient")

def main():
    """Main execution"""
    
    print("="*60)
    print("FEATURE SET COMPARISON ANALYSIS")
    print("="*60)
    print("\nComparing three feature configurations:")
    print("1. TF-IDF only (1000 features)")
    print("2. TF-IDF + LLM features (~1020 features)")
    print("3. TF-IDF + LLM + Market context (~1025 features)")
    print("="*60)
    
    # Initialize comparator
    comparator = FeatureSetComparator()
    
    # Run comparison
    results = comparator.run_comparison()
    
    # Create visualizations
    comparator.create_comparison_plots()
    comparator.create_feature_importance_plot()
    
    # Save results
    comparator.save_results()
    
    # Display summary
    comparator.display_summary()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - feature_set_comparison_results.csv")
    print("  - feature_comparison_summary.json")
    print("  - feature_set_comparison.png")
    print("  - feature_importance_comparison.png")

if __name__ == "__main__":
    main()