#!/usr/bin/env python3
"""
Train and Save Best Volatility Models
======================================
Trains the best performing models identified in our analysis:
1. MLP Regression for volatility magnitude (0.3611 correlation)
2. Random Forest Classification for high volatility events (0.394 F1 score)

Both use combined TF-IDF + LLM features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import hstack, csr_matrix
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING AND SAVING BEST VOLATILITY MODELS")
print("="*60)

# Load data
print("\n1. Loading data...")
data_path = '../../llm_features/filtered_data/'

train_df = pd.read_csv(f'{data_path}filtered_train.csv')
val_df = pd.read_csv(f'{data_path}filtered_val.csv')
test_df = pd.read_csv(f'{data_path}filtered_test.csv')

print(f"  Train: {len(train_df):,} samples")
print(f"  Val:   {len(val_df):,} samples")
print(f"  Test:  {len(test_df):,} samples")

# Prepare targets
print("\n2. Preparing targets...")

# Regression target: absolute returns (volatility)
y_train_reg = np.abs(train_df['adjusted_return_pct'].values)
y_val_reg = np.abs(val_df['adjusted_return_pct'].values)
y_test_reg = np.abs(test_df['adjusted_return_pct'].values)

# Classification target: high volatility binary (top 25%)
threshold = np.percentile(np.abs(train_df['adjusted_return_pct'].values), 75)
y_train_clf = (np.abs(train_df['adjusted_return_pct'].values) > threshold).astype(int)
y_val_clf = (np.abs(val_df['adjusted_return_pct'].values) > threshold).astype(int)
y_test_clf = (np.abs(test_df['adjusted_return_pct'].values) > threshold).astype(int)

print(f"  Volatility threshold (top 25%): {threshold:.3f}%")
print(f"  Train high vol events: {y_train_clf.mean():.1%}")
print(f"  Test high vol events: {y_test_clf.mean():.1%}")

# Create TF-IDF features
print("\n3. Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=10,
    max_df=0.7,
    stop_words='english',
    sublinear_tf=True,
    norm='l2'
)

X_train_tfidf = vectorizer.fit_transform(train_df['summary'].fillna(''))
X_val_tfidf = vectorizer.transform(val_df['summary'].fillna(''))
X_test_tfidf = vectorizer.transform(test_df['summary'].fillna(''))

print(f"  TF-IDF shape: {X_train_tfidf.shape}")

# Extract LLM features
print("\n4. Extracting LLM features...")

# Define feature groups
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

market_features = [
    'momentum_7d', 'momentum_30d', 'momentum_90d', 
    'momentum_365d', 'vix_level'
]

# Initialize label encoders
label_encoders = {}

def extract_llm_features(df, is_training=False):
    """Extract LLM features from dataframe"""
    features = []
    
    # Numeric features
    for feat in numeric_features:
        if feat in df.columns:
            features.append(df[feat].fillna(0).values.reshape(-1, 1))
    
    # Categorical features
    for feat in categorical_features:
        if feat in df.columns:
            if is_training:
                if feat not in label_encoders:
                    label_encoders[feat] = LabelEncoder()
                    label_encoders[feat].fit(df[feat].fillna('unknown').astype(str))
            
            if feat in label_encoders:
                encoded = []
                for val in df[feat].fillna('unknown').astype(str):
                    if val in label_encoders[feat].classes_:
                        encoded.append(label_encoders[feat].transform([val])[0])
                    else:
                        encoded.append(0)
                features.append(np.array(encoded).reshape(-1, 1))
    
    # Boolean features
    for feat in boolean_features:
        if feat in df.columns:
            features.append(df[feat].fillna(False).astype(int).values.reshape(-1, 1))
    
    # Market features
    for feat in market_features:
        if feat in df.columns:
            features.append(df[feat].fillna(0).values.reshape(-1, 1))
    
    if features:
        return np.hstack(features)
    return np.zeros((len(df), 1))

X_train_llm = extract_llm_features(train_df, is_training=True)
X_val_llm = extract_llm_features(val_df, is_training=False)
X_test_llm = extract_llm_features(test_df, is_training=False)

print(f"  LLM features shape: {X_train_llm.shape}")

# Scale LLM features
print("\n5. Scaling LLM features...")
scaler = StandardScaler(with_mean=False)
X_train_llm_scaled = scaler.fit_transform(X_train_llm)
X_val_llm_scaled = scaler.transform(X_val_llm)
X_test_llm_scaled = scaler.transform(X_test_llm)

# Combine features
print("\n6. Combining TF-IDF and LLM features...")
X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_llm_scaled)])
X_val_combined = hstack([X_val_tfidf, csr_matrix(X_val_llm_scaled)])
X_test_combined = hstack([X_test_tfidf, csr_matrix(X_test_llm_scaled)])

print(f"  Combined features shape: {X_train_combined.shape}")
print(f"  Total features: {X_train_combined.shape[1]:,}")

# Train MLP Regression Model
print("\n" + "="*60)
print("TRAINING MLP REGRESSION MODEL (Volatility Magnitude)")
print("="*60)

mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=128,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)

print("\nTraining MLP regressor...")
mlp_regressor.fit(X_train_combined, y_train_reg)

# Evaluate regression model
print("\nEvaluating regression model...")
test_pred_reg = mlp_regressor.predict(X_test_combined)
test_pred_reg = np.maximum(test_pred_reg, 0)  # Ensure non-negative

test_corr, test_p = pearsonr(y_test_reg, test_pred_reg)
test_spearman, _ = spearmanr(y_test_reg, test_pred_reg)
test_r2 = r2_score(y_test_reg, test_pred_reg)
test_mae = mean_absolute_error(y_test_reg, test_pred_reg)

print(f"\nRegression Model Performance:")
print(f"  Test Correlation: {test_corr:.4f} (p={test_p:.2e})")
print(f"  Test Spearman: {test_spearman:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test MAE: {test_mae:.3f}%")

# Train Random Forest Classification Model
print("\n" + "="*60)
print("TRAINING RANDOM FOREST CLASSIFICATION MODEL (High Volatility Events)")
print("="*60)

rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining Random Forest classifier...")
rf_classifier.fit(X_train_combined, y_train_clf)

# Evaluate classification model
print("\nEvaluating classification model...")
test_pred_clf = rf_classifier.predict(X_test_combined)
test_pred_proba = rf_classifier.predict_proba(X_test_combined)[:, 1]

accuracy = accuracy_score(y_test_clf, test_pred_clf)
precision = precision_score(y_test_clf, test_pred_clf)
recall = recall_score(y_test_clf, test_pred_clf)
f1 = f1_score(y_test_clf, test_pred_clf)
auc = roc_auc_score(y_test_clf, test_pred_proba)

print(f"\nClassification Model Performance:")
print(f"  Test Accuracy: {accuracy:.1%}")
print(f"  Test Precision: {precision:.1%}")
print(f"  Test Recall: {recall:.1%}")
print(f"  Test F1 Score: {f1:.3f}")
print(f"  Test AUC: {auc:.3f}")

# Save models and preprocessors
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

# Save regression model package
regression_package = {
    'model': mlp_regressor,
    'vectorizer': vectorizer,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'threshold': threshold,
    'performance': {
        'correlation': test_corr,
        'spearman': test_spearman,
        'r2': test_r2,
        'mae': test_mae
    },
    'metadata': {
        'created': datetime.now().isoformat(),
        'model_type': 'MLPRegressor',
        'features': 'combined_tfidf_llm',
        'n_features': X_train_combined.shape[1]
    }
}

joblib.dump(regression_package, 'model/best_volatility_model.pkl')
print("✓ Saved regression model to: best_volatility_model.pkl")

# Save classification model package
classification_package = {
    'model': rf_classifier,
    'vectorizer': vectorizer,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'threshold': threshold,
    'performance': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    },
    'metadata': {
        'created': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'features': 'combined_tfidf_llm',
        'n_features': X_train_combined.shape[1]
    }
}

joblib.dump(classification_package, 'model/best_classification_model.pkl')
print("✓ Saved classification model to: best_classification_model.pkl")

# Save a summary JSON for reference
summary = {
    'regression': {
        'model': 'MLPRegressor',
        'architecture': '256-128',
        'correlation': float(test_corr),
        'r2': float(test_r2),
        'mae': float(test_mae)
    },
    'classification': {
        'model': 'RandomForestClassifier',
        'n_estimators': 100,
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc': float(auc)
    },
    'features': {
        'tfidf_features': X_train_tfidf.shape[1],
        'llm_features': X_train_llm.shape[1],
        'total_features': X_train_combined.shape[1]
    },
    'data': {
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'volatility_threshold': float(threshold)
    },
    'created': datetime.now().isoformat()
}

with open('results/model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ Saved model summary to: model_summary.json")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print("\nModels saved successfully!")
print(f"  Regression: {test_corr:.4f} correlation")
print(f"  Classification: {f1:.3f} F1 score")
print("\nReady for use in trading strategies!")