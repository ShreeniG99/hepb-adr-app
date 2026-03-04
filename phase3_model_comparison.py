"""
Phase 3B: Multi-Model Comparison for Hepatitis B ADR Prediction
================================================================

This script trains and compares 5 machine learning models for predicting
5 adverse drug reaction (ADR) categories in Hepatitis B patients:
- Random Forest (baseline)
- XGBoost
- LightGBM
- Logistic Regression
- Stacking Ensemble (RF + XGB + LGBM with Logistic meta-learner)

Optimized for RECALL as the primary metric for clinical screening.

Author: ADR Prediction Pipeline
Date: 2024
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import xgboost as xgb
import lightgbm as lgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelComparisonConfig:
    """Configuration for model comparison pipeline"""

    # Paths
    DATA_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3_features"
    OUTPUT_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3b_ml_comparison"
    BASELINE_MODEL_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3_models"

    # Data file
    FEATURE_MATRIX_FILE = "feature_matrix.csv"

    # 5 ADR Categories
    ADR_CATEGORIES = [
        'hepatotoxicity',
        'nephrotoxicity',
        'bone_tooth',
        'hematologic',
        'neurologic'
    ]

    # Exact 26 features from baseline
    FEATURE_COLS = [
        # Demographics (6)
        'age_years', 'sex_binary', 'sex_female',
        'reporter_md', 'reporter_lawyer', 'reporter_consumer',
        # Severity (6)
        'outcome_death', 'outcome_life_threat', 'outcome_hospitalization',
        'outcome_disability', 'severe_disease', 'severity_score',
        # Polypharmacy (4)
        'drug_count', 'polypharmacy_low', 'polypharmacy_medium', 'polypharmacy_high',
        # Cohort (3)
        'cohort_hbv_mono', 'cohort_hiv_hbv', 'cohort_autoimmune',
        # Drug (7)
        'drug_entecavir', 'drug_tenofovir', 'drug_lamivudine',
        'drug_adefovir', 'drug_telbivudine', 'drug_interferon', 'drug_risk_score'
    ]

    # Training parameters (matching baseline exactly)
    RANDOM_STATE = 42
    TEST_SIZE = 0.2  # 80/20 split

    # Primary metric for model selection
    PRIMARY_METRIC = 'recall'


class PlotConfig:
    """Configuration for visualization"""
    DPI = 300
    FIGSIZE_LARGE = (16, 12)
    FIGSIZE_MEDIUM = (12, 8)
    FIGSIZE_SMALL = (10, 6)

    # Colorblind-friendly palette
    COLORS = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311']
    MODEL_COLORS = {
        'Random Forest': '#0077BB',
        'XGBoost': '#33BBEE',
        'LightGBM': '#009988',
        'Logistic Regression': '#EE7733',
        'Stacking Ensemble': '#CC3311'
    }


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging with file and console handlers"""

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training_log.log')

    logger = logging.getLogger('model_comparison')
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # File handler (detailed)
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))

    # Console handler (summary)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_feature_matrix(config: ModelComparisonConfig, logger: logging.Logger) -> pd.DataFrame:
    """
    Load feature matrix with memory optimization.

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        DataFrame with features and labels
    """
    filepath = os.path.join(config.DATA_DIR, config.FEATURE_MATRIX_FILE)
    logger.info(f"Loading feature matrix from {filepath}")

    # Define columns to load
    adr_cols = [f'adr_{cat}' for cat in config.ADR_CATEGORIES]
    usecols = ['PRIMARYID'] + config.FEATURE_COLS + adr_cols

    # Define dtypes for memory optimization
    dtype_mapping = {col: 'float32' for col in config.FEATURE_COLS}
    for col in adr_cols:
        dtype_mapping[col] = 'float32'

    df = pd.read_csv(filepath, usecols=usecols, dtype=dtype_mapping)

    logger.info(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def prepare_data(df: pd.DataFrame, config: ModelComparisonConfig, logger: logging.Logger) -> Tuple:
    """
    Prepare features and labels, perform train/test split.

    Args:
        df: Full feature matrix
        config: Configuration object
        logger: Logger instance

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    logger.info("Preparing features and labels...")

    # Extract features
    X = df[config.FEATURE_COLS].copy()

    # Fill missing age values with median
    if X['age_years'].isna().any():
        median_age = X['age_years'].median()
        X['age_years'] = X['age_years'].fillna(median_age)
        logger.info(f"Filled missing age values with median: {median_age:.1f}")

    # Extract labels
    adr_cols = [f'adr_{cat}' for cat in config.ADR_CATEGORIES]
    y = df[adr_cols].copy()

    # Create stratification variable (has any ADR)
    has_adr = (y.sum(axis=1) > 0).astype(int)

    logger.info(f"Class distribution - Has ADR: {has_adr.sum():,} ({has_adr.mean()*100:.1f}%)")

    # Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=has_adr
    )

    logger.info(f"Training set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    logger.info("Features scaled using StandardScaler")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_model_configs() -> Dict:
    """
    Get all model configurations.

    Returns:
        Dictionary mapping model names to their configurations
    """
    models = {}

    # Random Forest (baseline)
    models['Random Forest'] = {
        'short_name': 'RF',
        'estimator': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }

    # XGBoost
    models['XGBoost'] = {
        'short_name': 'XGB',
        'estimator': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        )
    }

    # LightGBM
    models['LightGBM'] = {
        'short_name': 'LGBM',
        'estimator': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }

    # Logistic Regression
    models['Logistic Regression'] = {
        'short_name': 'LR',
        'estimator': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            solver='lbfgs'
        )
    }

    # Stacking Ensemble
    base_estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=50, max_depth=8,
            class_weight='balanced', random_state=42, n_jobs=-1
        )),
        ('xgb', xgb.XGBClassifier(
            n_estimators=50, max_depth=5,
            scale_pos_weight=3, random_state=42, n_jobs=-1,
            eval_metric='logloss', verbosity=0
        )),
        ('lgbm', lgb.LGBMClassifier(
            n_estimators=50, max_depth=5,
            class_weight='balanced', random_state=42,
            n_jobs=-1, verbose=-1
        ))
    ]

    models['Stacking Ensemble'] = {
        'short_name': 'Stack',
        'estimator': StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            cv=3,
            n_jobs=-1
        )
    }

    return models


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(
    model_name: str,
    model_config: Dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[MultiOutputClassifier, float]:
    """
    Train a single model using MultiOutputClassifier wrapper.

    Args:
        model_name: Name of the model
        model_config: Model configuration dict
        X_train: Training features
        y_train: Training labels
        logger: Logger instance

    Returns:
        Trained model and training time in seconds
    """
    logger.info(f"Training {model_name}...")

    # Wrap in MultiOutputClassifier
    multi_model = MultiOutputClassifier(model_config['estimator'], n_jobs=-1)

    # Train with timing
    start_time = time.time()
    multi_model.fit(X_train, y_train)
    training_time = time.time() - start_time

    logger.info(f"{model_name} trained in {training_time:.2f} seconds")

    return multi_model, training_time


def train_all_models(
    model_configs: Dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[Dict, Dict]:
    """
    Train all models with progress tracking.

    Args:
        model_configs: Dictionary of model configurations
        X_train: Training features
        y_train: Training labels
        logger: Logger instance

    Returns:
        Dictionary of trained models and training times
    """
    trained_models = {}
    training_times = {}

    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    for model_name, model_config in tqdm(model_configs.items(), desc="Training Models"):
        model, train_time = train_model(
            model_name, model_config, X_train, y_train, logger
        )
        trained_models[model_name] = model
        training_times[model_name] = train_time

    print("\nTraining complete!")
    print("-"*60)
    for name, t in training_times.items():
        print(f"  {name}: {t:.2f}s")

    return trained_models, training_times


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    model: MultiOutputClassifier,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelComparisonConfig,
    logger: logging.Logger
) -> Dict:
    """
    Evaluate a single model across all ADR categories.

    Args:
        model: Trained model
        model_name: Name of the model
        X_test: Test features
        y_test: Test labels
        config: Configuration object
        logger: Logger instance

    Returns:
        Dictionary with metrics per ADR category
    """
    logger.debug(f"Evaluating {model_name}...")

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = np.array([est.predict_proba(X_test)[:, 1]
                             for est in model.estimators_]).T

    results = {}

    for i, category in enumerate(config.ADR_CATEGORIES):
        y_true = y_test.iloc[:, i]
        y_p = y_pred[:, i]
        y_proba = y_pred_proba[:, i]

        results[category] = {
            'accuracy': accuracy_score(y_true, y_p),
            'precision': precision_score(y_true, y_p, zero_division=0),
            'recall': recall_score(y_true, y_p, zero_division=0),
            'f1': f1_score(y_true, y_p, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_p)
        }

    # Calculate mean metrics
    results['mean'] = {
        metric: np.mean([results[cat][metric] for cat in config.ADR_CATEGORIES])
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    }

    return results


def evaluate_all_models(
    trained_models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelComparisonConfig,
    logger: logging.Logger
) -> Dict:
    """
    Evaluate all trained models.

    Args:
        trained_models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        config: Configuration object
        logger: Logger instance

    Returns:
        Dictionary with results for all models
    """
    all_results = {}

    print("\n" + "="*60)
    print("EVALUATING MODELS")
    print("="*60)

    for model_name, model in tqdm(trained_models.items(), desc="Evaluating Models"):
        all_results[model_name] = evaluate_model(
            model, model_name, X_test, y_test, config, logger
        )

    return all_results


def get_roc_curves_data(
    trained_models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelComparisonConfig
) -> Dict:
    """
    Get ROC curve data for all models and ADR categories.

    Returns:
        Dictionary with FPR, TPR, and AUC for each model/category combination
    """
    roc_data = {}

    for model_name, model in trained_models.items():
        roc_data[model_name] = {}
        y_pred_proba = np.array([est.predict_proba(X_test)[:, 1]
                                 for est in model.estimators_]).T

        for i, category in enumerate(config.ADR_CATEGORIES):
            y_true = y_test.iloc[:, i]
            y_proba = y_pred_proba[:, i]

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)

            roc_data[model_name][category] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': auc
            }

    return roc_data


# =============================================================================
# RESULTS GENERATION
# =============================================================================

def generate_comparison_tables(
    all_results: Dict,
    training_times: Dict,
    config: ModelComparisonConfig,
    output_dir: str,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate comparison tables and save to CSV.

    Returns:
        Full results DataFrame and summary DataFrame
    """
    # Full results table
    rows = []
    for model_name, results in all_results.items():
        for category in config.ADR_CATEGORIES:
            row = {
                'Model': model_name,
                'ADR_Category': category,
                'Accuracy': results[category]['accuracy'],
                'Precision': results[category]['precision'],
                'Recall': results[category]['recall'],
                'F1_Score': results[category]['f1'],
                'AUC_ROC': results[category]['auc_roc']
            }
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(output_dir, 'model_comparison_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Full results saved to {results_path}")

    # Summary table (mean across ADR categories)
    summary_rows = []
    for model_name, results in all_results.items():
        row = {
            'Model': model_name,
            'Mean_Accuracy': results['mean']['accuracy'],
            'Mean_Precision': results['mean']['precision'],
            'Mean_Recall': results['mean']['recall'],
            'Mean_F1_Score': results['mean']['f1'],
            'Mean_AUC_ROC': results['mean']['auc_roc'],
            'Training_Time_Sec': training_times[model_name]
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Mean_Recall', ascending=False)
    summary_path = os.path.join(output_dir, 'model_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved to {summary_path}")

    return results_df, summary_df


def select_best_model(
    all_results: Dict,
    trained_models: Dict,
    config: ModelComparisonConfig,
    logger: logging.Logger
) -> Tuple[str, object, Dict]:
    """
    Select the best model based on mean recall.

    Returns:
        Best model name, best model object, and its metrics
    """
    best_name = None
    best_recall = -1

    for model_name, results in all_results.items():
        mean_recall = results['mean']['recall']
        if mean_recall > best_recall:
            best_recall = mean_recall
            best_name = model_name

    logger.info(f"Best model by recall: {best_name} (Mean Recall: {best_recall:.4f})")

    return best_name, trained_models[best_name], all_results[best_name]


# =============================================================================
# VISUALIZATION
# =============================================================================

def setup_plot_style():
    """Configure matplotlib/seaborn for publication quality"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(PlotConfig.COLORS)

    plt.rcParams.update({
        'figure.dpi': PlotConfig.DPI,
        'savefig.dpi': PlotConfig.DPI,
        'savefig.bbox': 'tight',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })


def plot_confusion_matrices(
    model: MultiOutputClassifier,
    model_name: str,
    short_name: str,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelComparisonConfig,
    output_dir: str
):
    """Plot confusion matrices for all ADR categories"""
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(2, 3, figsize=PlotConfig.FIGSIZE_LARGE)
    axes = axes.flatten()

    for i, category in enumerate(config.ADR_CATEGORIES):
        cm = confusion_matrix(y_test.iloc[:, i], y_pred[:, i])

        # Normalize for display
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=['No ADR', 'ADR'],
            yticklabels=['No ADR', 'ADR'],
            ax=axes[i]
        )
        axes[i].set_title(f'{category.replace("_", " ").title()}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Hide the 6th subplot
    axes[5].axis('off')

    plt.suptitle(f'{model_name} - Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'confusion_matrices_{short_name}.png')
    plt.savefig(save_path, dpi=PlotConfig.DPI)
    plt.close()


def plot_roc_curves_comparison(
    roc_data: Dict,
    config: ModelComparisonConfig,
    output_dir: str
):
    """Plot ROC curves comparing all models for each ADR category"""
    fig, axes = plt.subplots(2, 3, figsize=PlotConfig.FIGSIZE_LARGE)
    axes = axes.flatten()

    for i, category in enumerate(config.ADR_CATEGORIES):
        ax = axes[i]

        for model_name, data in roc_data.items():
            cat_data = data[category]
            color = PlotConfig.MODEL_COLORS[model_name]
            ax.plot(
                cat_data['fpr'], cat_data['tpr'],
                color=color,
                label=f"{model_name} (AUC={cat_data['auc']:.3f})",
                linewidth=2
            )

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{category.replace("_", " ").title()}')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

    # Hide the 6th subplot
    axes[5].axis('off')

    plt.suptitle('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'roc_curves_comparison.png')
    plt.savefig(save_path, dpi=PlotConfig.DPI)
    plt.close()


def plot_metrics_comparison_bar(
    summary_df: pd.DataFrame,
    output_dir: str
):
    """Plot grouped bar chart comparing models across metrics"""
    metrics = ['Mean_Accuracy', 'Mean_Precision', 'Mean_Recall', 'Mean_F1_Score', 'Mean_AUC_ROC']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']

    fig, ax = plt.subplots(figsize=PlotConfig.FIGSIZE_MEDIUM)

    x = np.arange(len(summary_df))
    width = 0.15

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset,
            summary_df[metric],
            width,
            label=label,
            color=PlotConfig.COLORS[i]
        )

    ax.set_ylabel('Score')
    ax.set_xlabel('Model')
    ax.set_title('Model Comparison - All Metrics', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.884, color='red', linestyle='--', alpha=0.7, label='Baseline Recall')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_comparison_bar.png')
    plt.savefig(save_path, dpi=PlotConfig.DPI)
    plt.close()


def plot_recall_heatmap(
    results_df: pd.DataFrame,
    output_dir: str
):
    """Plot heatmap of recall scores: Models x ADR Categories"""
    pivot = results_df.pivot(index='Model', columns='ADR_Category', values='Recall')

    # Reorder columns
    pivot = pivot[['hepatotoxicity', 'nephrotoxicity', 'bone_tooth', 'hematologic', 'neurologic']]

    fig, ax = plt.subplots(figsize=PlotConfig.FIGSIZE_SMALL)

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        ax=ax,
        cbar_kws={'label': 'Recall Score'}
    )

    ax.set_title('Recall Scores by Model and ADR Category', fontweight='bold')
    ax.set_xlabel('ADR Category')
    ax.set_ylabel('Model')

    # Format x-axis labels
    ax.set_xticklabels([label.get_text().replace('_', ' ').title()
                        for label in ax.get_xticklabels()], rotation=45, ha='right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'recall_heatmap.png')
    plt.savefig(save_path, dpi=PlotConfig.DPI)
    plt.close()


def generate_all_plots(
    trained_models: Dict,
    model_configs: Dict,
    all_results: Dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    config: ModelComparisonConfig,
    output_dir: str,
    logger: logging.Logger
):
    """Generate all visualization plots"""
    setup_plot_style()

    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Confusion matrices for each model
    for model_name, model in tqdm(trained_models.items(), desc="Confusion Matrices"):
        short_name = model_configs[model_name]['short_name']
        plot_confusion_matrices(
            model, model_name, short_name, X_test, y_test, config, output_dir
        )

    # ROC curves comparison
    print("Generating ROC curves comparison...")
    roc_data = get_roc_curves_data(trained_models, X_test, y_test, config)
    plot_roc_curves_comparison(roc_data, config, output_dir)

    # Metrics comparison bar chart
    print("Generating metrics comparison bar chart...")
    plot_metrics_comparison_bar(summary_df, output_dir)

    # Recall heatmap
    print("Generating recall heatmap...")
    plot_recall_heatmap(results_df, output_dir)

    logger.info("All visualizations generated successfully")


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_comparison_report(
    summary_df: pd.DataFrame,
    all_results: Dict,
    best_model_name: str,
    training_times: Dict,
    config: ModelComparisonConfig,
    output_dir: str,
    logger: logging.Logger
):
    """Generate comprehensive text report"""
    report_path = os.path.join(output_dir, 'comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("HEPATITIS B ADR PREDICTION - MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Primary Optimization Metric: {config.PRIMARY_METRIC.upper()}\n\n")

        # Baseline comparison
        f.write("-"*80 + "\n")
        f.write("BASELINE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write("Previous RF Baseline: 88.4% Recall, 33.1% Precision, 0.922 AUC\n\n")

        # Model ranking
        f.write("-"*80 + "\n")
        f.write("MODEL RANKING (by Mean Recall)\n")
        f.write("-"*80 + "\n\n")

        for idx, row in summary_df.iterrows():
            f.write(f"#{summary_df.index.get_loc(idx)+1}: {row['Model']}\n")
            f.write(f"    Mean Recall:    {row['Mean_Recall']:.4f}\n")
            f.write(f"    Mean Precision: {row['Mean_Precision']:.4f}\n")
            f.write(f"    Mean F1 Score:  {row['Mean_F1_Score']:.4f}\n")
            f.write(f"    Mean AUC-ROC:   {row['Mean_AUC_ROC']:.4f}\n")
            f.write(f"    Training Time:  {row['Training_Time_Sec']:.2f}s\n\n")

        # Best model
        f.write("-"*80 + "\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write("-"*80 + "\n\n")

        best_results = all_results[best_model_name]
        f.write("Performance by ADR Category:\n\n")

        for category in config.ADR_CATEGORIES:
            cat_results = best_results[category]
            f.write(f"  {category.replace('_', ' ').title()}:\n")
            f.write(f"    Accuracy:  {cat_results['accuracy']:.4f}\n")
            f.write(f"    Precision: {cat_results['precision']:.4f}\n")
            f.write(f"    Recall:    {cat_results['recall']:.4f}\n")
            f.write(f"    F1 Score:  {cat_results['f1']:.4f}\n")
            f.write(f"    AUC-ROC:   {cat_results['auc_roc']:.4f}\n\n")

        # Conclusion
        f.write("-"*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("-"*80 + "\n\n")

        best_recall = best_results['mean']['recall']
        baseline_recall = 0.884

        if best_recall > baseline_recall:
            improvement = (best_recall - baseline_recall) / baseline_recall * 100
            f.write(f"The {best_model_name} achieved {best_recall:.4f} mean recall,\n")
            f.write(f"representing a {improvement:.1f}% improvement over the baseline.\n")
        else:
            f.write(f"The {best_model_name} achieved {best_recall:.4f} mean recall.\n")
            f.write("Consider hyperparameter tuning for further improvement.\n")

        f.write("\n" + "="*80 + "\n")

    logger.info(f"Report saved to {report_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_model_comparison_pipeline():
    """
    Main orchestration function for model comparison.
    """
    config = ModelComparisonConfig()

    # Setup
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(config.OUTPUT_DIR)

    print("\n" + "="*80)
    print("HEPATITIS B ADR PREDICTION - MODEL COMPARISON PIPELINE")
    print("="*80)
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print("="*80 + "\n")

    logger.info("Starting model comparison pipeline")

    try:
        # 1. Load data
        df = load_feature_matrix(config, logger)

        # 2. Prepare train/test split
        X_train, X_test, y_train, y_test, scaler = prepare_data(df, config, logger)

        # Free memory
        del df

        # 3. Save scaler
        scaler_path = os.path.join(config.OUTPUT_DIR, 'feature_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # 4. Get model configurations
        model_configs = get_model_configs()

        # 5. Train all models
        trained_models, training_times = train_all_models(
            model_configs, X_train, y_train, logger
        )

        # 6. Evaluate all models
        all_results = evaluate_all_models(
            trained_models, X_test, y_test, config, logger
        )

        # 7. Generate comparison tables
        results_df, summary_df = generate_comparison_tables(
            all_results, training_times, config, config.OUTPUT_DIR, logger
        )

        # 8. Select best model
        best_name, best_model, best_metrics = select_best_model(
            all_results, trained_models, config, logger
        )

        # 9. Save best model
        best_model_path = os.path.join(config.OUTPUT_DIR, 'best_model_recall_optimized.pkl')
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")

        # 10. Generate visualizations
        generate_all_plots(
            trained_models, model_configs, all_results,
            X_test, y_test, results_df, summary_df,
            config, config.OUTPUT_DIR, logger
        )

        # 11. Generate report
        generate_comparison_report(
            summary_df, all_results, best_name, training_times,
            config, config.OUTPUT_DIR, logger
        )

        # Print summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nBest Model: {best_name}")
        print(f"Mean Recall: {best_metrics['mean']['recall']:.4f}")
        print(f"Mean Precision: {best_metrics['mean']['precision']:.4f}")
        print(f"Mean AUC-ROC: {best_metrics['mean']['auc_roc']:.4f}")
        print(f"\nAll outputs saved to: {config.OUTPUT_DIR}")
        print("="*80 + "\n")

        logger.info("Pipeline completed successfully")

        return trained_models, all_results, summary_df

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_model_comparison_pipeline()
