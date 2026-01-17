# phase3_model_training.py

"""
Phase 3B: ML Model Training
Train multi-label classifier to predict ADR categories

Model: Random Forest (interpretable, works with small datasets)
Task: Multi-label classification (5 ADR categories)
Evaluation: ROC-AUC, Precision-Recall, F1-score per category
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

class ModelConfig:
    """Configuration for model training"""
    
    # Paths
    DATA_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3_features"
    OUTPUT_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3_models"
    
    # ADR categories (must match feature extraction)
    ADR_CATEGORIES = [
        'hepatotoxicity',
        'nephrotoxicity',
        'bone_tooth',
        'hematologic',
        'neurologic'
    ]
    
    # Feature columns to use
    FEATURE_COLS = [
        # Demographics
        'age_years', 'sex_binary', 'sex_female',
        'reporter_md', 'reporter_lawyer', 'reporter_consumer',
        
        # Severity
        'outcome_death', 'outcome_life_threat', 'outcome_hospitalization',
        'outcome_disability', 'severe_disease', 'severity_score',
        
        # Polypharmacy
        'drug_count', 'polypharmacy_low', 'polypharmacy_medium', 'polypharmacy_high',
        
        # Cohort
        'cohort_hbv_mono', 'cohort_hiv_hbv', 'cohort_autoimmune',
        
        # Drug
        'drug_entecavir', 'drug_tenofovir', 'drug_lamivudine', 
        'drug_adefovir', 'drug_telbivudine', 'drug_interferon', 'drug_risk_score'
    ]
    
    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    
    # Random Forest parameters (optimized for small dataset)
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    MIN_SAMPLES_SPLIT = 50
    MIN_SAMPLES_LEAF = 20

config = ModelConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_feature_matrix():
    """Load the feature matrix from Phase 3A"""
    print("\n" + "="*80)
    print("LOADING FEATURE MATRIX")
    print("="*80)
    
    filepath = os.path.join(config.DATA_DIR, 'feature_matrix.csv')
    
    if not os.path.exists(filepath):
        print(f"  ✗ Feature matrix not found: {filepath}")
        print(f"  Run phase3_feature_extraction.py first!")
        return None
    
    # Check file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    
    print(f"  Loading data...")
    df = pd.read_csv(filepath)
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    print(f"  ✓ Loaded feature matrix: {df.shape}")
    print(f"    Patients: {len(df):,}")
    print(f"    Total columns: {df.shape[1]}")
    print(f"    Memory usage: {memory_mb:.1f} MB")
    
    return df

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df):
    """
    Prepare features (X) and labels (y) for training
    """
    print("\n" + "="*80)
    print("PREPARING DATA")
    print("="*80)
    
    # Extract features
    print(f"\n  Selecting {len(config.FEATURE_COLS)} features...")
    X = df[config.FEATURE_COLS].copy()
    
    # Handle missing values in age
    X['age_years'] = X['age_years'].fillna(X['age_years'].median())
    
    # Extract multi-label targets
    y_cols = [f'adr_{cat}' for cat in config.ADR_CATEGORIES]
    y = df[y_cols].copy()
    
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    
    # Print label distribution
    print(f"\n  ADR Category Prevalence:")
    for i, cat in enumerate(config.ADR_CATEGORIES):
        prevalence = y.iloc[:, i].mean()
        count = y.iloc[:, i].sum()
        print(f"    {cat:<20} {prevalence:>6.1%} ({count:>6,} patients)")
    
    # Check for samples with no labels
    no_labels = (y.sum(axis=1) == 0).sum()
    print(f"\n  Samples with no ADR labels: {no_labels:,}")
    
    # Feature scaling
    print(f"\n  Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Save scaler
    scaler_path = os.path.join(config.OUTPUT_DIR, 'feature_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved scaler: {scaler_path}")
    
    return X_scaled, y, scaler


def split_data(X, y):
    """
    Split data into train and test sets
    Use stratified split based on at least one ADR present
    """
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    
    # Create stratification variable (has any ADR)
    has_adr = (y.sum(axis=1) > 0).astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=has_adr
    )
    
    print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    print(f"\n  Train ADR prevalence:")
    for i, cat in enumerate(config.ADR_CATEGORIES):
        prevalence = y_train.iloc[:, i].mean()
        print(f"    {cat:<20} {prevalence:>6.1%}")
    
    print(f"\n  Test ADR prevalence:")
    for i, cat in enumerate(config.ADR_CATEGORIES):
        prevalence = y_test.iloc[:, i].mean()
        print(f"    {cat:<20} {prevalence:>6.1%}")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train):
    """
    Train multi-label Random Forest classifier
    """
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    print(f"\n  Model: Multi-label Random Forest")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Target categories: {len(config.ADR_CATEGORIES)}")
    
    print(f"\n  Hyperparameters:")
    print(f"    - n_estimators: {config.N_ESTIMATORS}")
    print(f"    - max_depth: {config.MAX_DEPTH}")
    print(f"    - min_samples_split: {config.MIN_SAMPLES_SPLIT}")
    print(f"    - min_samples_leaf: {config.MIN_SAMPLES_LEAF}")
    print(f"    - class_weight: balanced")
    
    # Define base classifier
    base_clf = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',  # Handle imbalance
        verbose=0
    )
    
    # Multi-label wrapper
    model = MultiOutputClassifier(base_clf, n_jobs=-1)
    
    # Train
    print(f"\n  Training in progress...")
    print(f"  (This may take 5-15 minutes depending on your hardware)")
    
    import time
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Training complete!")
    print(f"  Training time: {elapsed/60:.1f} minutes")
    
    # Save model
    model_path = os.path.join(config.OUTPUT_DIR, 'adr_classifier.pkl')
    joblib.dump(model, model_path)
    
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  ✓ Saved model: {model_path}")
    print(f"  Model file size: {model_size_mb:.1f} MB")
    
    return model

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Comprehensive model evaluation
    """
    print("\n" + "="*80)
    print("EVALUATING MODEL")
    print("="*80)
    
    # Predictions
    print(f"\n  Making predictions on train set...")
    y_pred_train = model.predict(X_train)
    
    print(f"  Making predictions on test set...")
    y_pred_test = model.predict(X_test)
    
    print(f"  Calculating prediction probabilities...")
    y_pred_proba_test = np.array([est.predict_proba(X_test)[:, 1] for est in model.estimators_]).T
    
    # Metrics per category
    results = []
    
    print(f"\n  Performance by ADR Category:")
    print(f"  {'':<20} {'Train AUC':<12} {'Test AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("  " + "-"*88)
    
    for i, cat in enumerate(config.ADR_CATEGORIES):
        # ROC-AUC
        try:
            train_auc = roc_auc_score(y_train.iloc[:, i], y_pred_train[:, i])
            test_auc = roc_auc_score(y_test.iloc[:, i], y_pred_proba_test[:, i])
        except:
            train_auc = test_auc = 0.0
        
        # Precision, Recall, F1
        precision = precision_score(y_test.iloc[:, i], y_pred_test[:, i], zero_division=0)
        recall = recall_score(y_test.iloc[:, i], y_pred_test[:, i], zero_division=0)
        f1 = f1_score(y_test.iloc[:, i], y_pred_test[:, i], zero_division=0)
        
        print(f"  {cat:<20} {train_auc:<12.3f} {test_auc:<12.3f} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")
        
        results.append({
            'category': cat,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Overall metrics
    mean_test_auc = np.mean([r['test_auc'] for r in results])
    mean_f1 = np.mean([r['f1'] for r in results])
    
    print(f"\n  Overall Performance:")
    print(f"    Mean Test AUC: {mean_test_auc:.3f}")
    print(f"    Mean F1 Score: {mean_f1:.3f}")
    
    # Interpretation guide
    print(f"\n  Performance Interpretation:")
    if mean_test_auc >= 0.75:
        print(f"    ✓ EXCELLENT: AUC {mean_test_auc:.3f} - Model is highly predictive")
    elif mean_test_auc >= 0.70:
        print(f"    ✓ GOOD: AUC {mean_test_auc:.3f} - Model performs well (expected for FAERS)")
    elif mean_test_auc >= 0.65:
        print(f"    ⚠ FAIR: AUC {mean_test_auc:.3f} - Model has some predictive power")
    else:
        print(f"    ⚠ WEAK: AUC {mean_test_auc:.3f} - Model needs improvement")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(config.OUTPUT_DIR, 'model_performance.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n  ✓ Saved performance metrics: {results_path}")
    
    return results_df, y_pred_proba_test


def plot_roc_curves(y_test, y_pred_proba):
    """
    Plot ROC curves for each ADR category
    """
    print("\n  Generating ROC curves...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, cat in enumerate(config.ADR_CATEGORIES):
        try:
            fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_test.iloc[:, i], y_pred_proba[:, i])
            
            axes[i].plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}', color='#2ecc71')
            axes[i].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            axes[i].set_xlabel('False Positive Rate', fontweight='bold')
            axes[i].set_ylabel('True Positive Rate', fontweight='bold')
            axes[i].set_title(f'{cat.replace("_", " ").title()}', fontweight='bold', fontsize=12)
            axes[i].legend(loc='lower right', frameon=True)
            axes[i].grid(alpha=0.3)
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])
        except:
            axes[i].text(0.5, 0.5, 'Insufficient data', 
                        ha='center', va='center', fontsize=12)
            axes[i].set_title(f'{cat.replace("_", " ").title()}', fontweight='bold', fontsize=12)
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.suptitle('ROC Curves by ADR Category', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(config.OUTPUT_DIR, 'roc_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved ROC curves: {plot_path}")


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for each ADR category
    """
    print("\n  Generating feature importance plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, cat in enumerate(config.ADR_CATEGORIES):
        # Get feature importances from the i-th classifier
        importances = model.estimators_[i].feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:10]  # Top 10
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
        axes[i].barh(range(10), importances[indices], color=colors)
        axes[i].set_yticks(range(10))
        axes[i].set_yticklabels([feature_names[idx] for idx in indices], fontsize=9)
        axes[i].set_xlabel('Importance', fontweight='bold')
        axes[i].set_title(f'{cat.replace("_", " ").title()}', fontweight='bold', fontsize=12)
        axes[i].invert_yaxis()
        axes[i].grid(axis='x', alpha=0.3)
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.suptitle('Top 10 Feature Importances by ADR Category', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(config.OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved feature importance: {plot_path}")


def generate_evaluation_report(model, X_test, y_test, results_df):
    """
    Generate comprehensive evaluation report
    """
    report = []
    report.append("\n" + "="*80)
    report.append("MODEL EVALUATION REPORT")
    report.append("="*80 + "\n")
    
    report.append(f"Model Type: Multi-label Random Forest")
    report.append(f"Test Set Size: {len(X_test):,} patients")
    report.append(f"Number of Features: {X_test.shape[1]}")
    report.append(f"Number of ADR Categories: {len(config.ADR_CATEGORIES)}\n")
    
    report.append("HYPERPARAMETERS:")
    report.append(f"  - n_estimators: {config.N_ESTIMATORS}")
    report.append(f"  - max_depth: {config.MAX_DEPTH}")
    report.append(f"  - min_samples_split: {config.MIN_SAMPLES_SPLIT}")
    report.append(f"  - min_samples_leaf: {config.MIN_SAMPLES_LEAF}")
    report.append(f"  - class_weight: balanced\n")
    
    report.append("PERFORMANCE BY CATEGORY:")
    report.append(results_df.to_string(index=False))
    
    report.append("\n\nOVERALL METRICS:")
    report.append(f"  Mean Test AUC: {results_df['test_auc'].mean():.3f}")
    report.append(f"  Mean Precision: {results_df['precision'].mean():.3f}")
    report.append(f"  Mean Recall: {results_df['recall'].mean():.3f}")
    report.append(f"  Mean F1 Score: {results_df['f1'].mean():.3f}")
    
    report.append("\n\nINTERPRETATION:")
    mean_auc = results_df['test_auc'].mean()
    if mean_auc >= 0.75:
        report.append("  Performance: EXCELLENT")
        report.append("  The model shows strong predictive ability across all ADR categories.")
    elif mean_auc >= 0.70:
        report.append("  Performance: GOOD")
        report.append("  The model performs well given FAERS data limitations.")
        report.append("  This is typical for real-world pharmacovigilance applications.")
    elif mean_auc >= 0.65:
        report.append("  Performance: FAIR")
        report.append("  The model has some predictive power but could benefit from:")
        report.append("    - Additional clinical features (lab values)")
        report.append("    - Larger sample size")
        report.append("    - Feature engineering")
    else:
        report.append("  Performance: WEAK")
        report.append("  Consider revisiting feature selection and model architecture.")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save report
    report_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n✓ Evaluation report saved: {report_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_training_pipeline():
    """Main training pipeline"""
    
    print("\n" + "🚀"*40)
    print("PHASE 3B: MODEL TRAINING")
    print("🚀"*40)
    
    # Load data
    df = load_feature_matrix()
    if df is None:
        return None
    
    # Prepare data
    X, y, scaler = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    results_df, y_pred_proba = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Generate visualizations
    plot_roc_curves(y_test, y_pred_proba)
    plot_feature_importance(model, X.columns.tolist())
    
    # Generate report
    generate_evaluation_report(model, X_test, y_test, results_df)
    
    print("\n" + "✅"*40)
    print("MODEL TRAINING COMPLETE!")
    print("✅"*40)
    print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")
    
    print(f"\n📁 Generated Files:")
    print(f"  ✓ adr_classifier.pkl - Trained model")
    print(f"  ✓ feature_scaler.pkl - Feature scaler")
    print(f"  ✓ model_performance.csv - Performance metrics")
    print(f"  ✓ roc_curves.png - ROC curve visualization")
    print(f"  ✓ feature_importance.png - Feature importance plots")
    print(f"  ✓ evaluation_report.txt - Full evaluation report")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Review ROC curves and feature importance plots")
    print(f"  2. Check if test AUC is 0.65-0.80 (expected range)")
    print(f"  3. Ready to build web app (Phase 4)!")
    
    return model, scaler, results_df


if __name__ == "__main__":
    model, scaler, results = run_training_pipeline()
