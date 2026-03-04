"""
Phase 3B: SHAP Explainability Analysis for Hepatitis B ADR Prediction
======================================================================

This script performs SHAP (SHapley Additive exPlanations) analysis on the
best-performing ADR prediction model to provide:
- Global feature importance (Summary plots)
- Local explanations (Waterfall plots for high-risk cases)
- Feature interactions (Dependence plots)
- Clinical interpretation report

Author: ADR Prediction Pipeline
Date: 2024
"""

import os
import sys
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

import shap
from sklearn.multioutput import MultiOutputClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class SHAPConfig:
    """Configuration for SHAP analysis"""

    # Paths
    MODEL_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3b_ml_comparison"
    DATA_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3_features"
    OUTPUT_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3b_ml_comparison\shap_analysis"

    # Model files
    MODEL_FILE = "best_model_recall_optimized.pkl"
    SCALER_FILE = "feature_scaler.pkl"
    FEATURE_MATRIX_FILE = "feature_matrix.csv"

    # SHAP computation settings
    SHAP_SAMPLE_SIZE = 5000  # Sample for faster computation
    SHAP_BACKGROUND_SIZE = 100  # Background samples for KernelExplainer

    # High-risk case selection
    N_HIGH_RISK_CASES = 3  # Number of waterfall plots per ADR

    # Dependence plot settings
    N_TOP_FEATURES = 5  # Features for dependence plots

    # 5 ADR Categories
    ADR_CATEGORIES = [
        'hepatotoxicity',
        'nephrotoxicity',
        'bone_tooth',
        'hematologic',
        'neurologic'
    ]

    # Exact 26 features (must match model training)
    FEATURE_COLS = [
        'age_years', 'sex_binary', 'sex_female',
        'reporter_md', 'reporter_lawyer', 'reporter_consumer',
        'outcome_death', 'outcome_life_threat', 'outcome_hospitalization',
        'outcome_disability', 'severe_disease', 'severity_score',
        'drug_count', 'polypharmacy_low', 'polypharmacy_medium', 'polypharmacy_high',
        'cohort_hbv_mono', 'cohort_hiv_hbv', 'cohort_autoimmune',
        'drug_entecavir', 'drug_tenofovir', 'drug_lamivudine',
        'drug_adefovir', 'drug_telbivudine', 'drug_interferon', 'drug_risk_score'
    ]

    # Clinical-friendly feature names for reports
    FEATURE_DISPLAY_NAMES = {
        'age_years': 'Patient Age (years)',
        'sex_binary': 'Male Sex',
        'sex_female': 'Female Sex',
        'reporter_md': 'Healthcare Provider Report',
        'reporter_lawyer': 'Legal Representative Report',
        'reporter_consumer': 'Consumer Report',
        'outcome_death': 'Prior Death Outcome',
        'outcome_life_threat': 'Prior Life-Threatening Event',
        'outcome_hospitalization': 'Prior Hospitalization',
        'outcome_disability': 'Prior Disability',
        'severe_disease': 'Severe Disease History',
        'severity_score': 'Composite Severity Score',
        'drug_count': 'Total Concurrent Medications',
        'polypharmacy_low': 'Low Polypharmacy (1-2 drugs)',
        'polypharmacy_medium': 'Medium Polypharmacy (3-5 drugs)',
        'polypharmacy_high': 'High Polypharmacy (6+ drugs)',
        'cohort_hbv_mono': 'HBV Mono-infection',
        'cohort_hiv_hbv': 'HIV/HBV Co-infection',
        'cohort_autoimmune': 'Autoimmune Co-morbidity',
        'drug_entecavir': 'Entecavir Use',
        'drug_tenofovir': 'Tenofovir Use',
        'drug_lamivudine': 'Lamivudine Use',
        'drug_adefovir': 'Adefovir Use',
        'drug_telbivudine': 'Telbivudine Use',
        'drug_interferon': 'Interferon Use',
        'drug_risk_score': 'Drug Risk Score (ROR)'
    }

    # ADR display names
    ADR_DISPLAY_NAMES = {
        'hepatotoxicity': 'Hepatotoxicity (Liver Damage)',
        'nephrotoxicity': 'Nephrotoxicity (Kidney Damage)',
        'bone_tooth': 'Bone/Dental Complications',
        'hematologic': 'Hematologic Toxicity (Blood Disorders)',
        'neurologic': 'Neurologic Toxicity (Nerve Damage)'
    }

    # Random state for reproducibility
    RANDOM_STATE = 42
    TEST_SIZE = 0.2


class PlotConfig:
    """Configuration for visualization"""
    DPI = 300
    FIGSIZE_LARGE = (16, 12)
    FIGSIZE_MEDIUM = (14, 10)
    FIGSIZE_SMALL = (10, 8)
    FIGSIZE_WATERFALL = (12, 6)

    # Color schemes
    SHAP_CMAP = 'coolwarm'


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging with file and console handlers"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'shap_analysis_log.log')

    logger = logging.getLogger('shap_analysis')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# DATA AND MODEL LOADING
# =============================================================================

def load_model_and_scaler(config: SHAPConfig, logger: logging.Logger) -> Tuple:
    """
    Load the best model and feature scaler.

    Returns:
        model, scaler
    """
    model_path = os.path.join(config.MODEL_DIR, config.MODEL_FILE)
    scaler_path = os.path.join(config.MODEL_DIR, config.SCALER_FILE)

    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    return model, scaler


def load_and_prepare_data(
    config: SHAPConfig,
    scaler,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load feature matrix and prepare train/test split matching model training.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    filepath = os.path.join(config.DATA_DIR, config.FEATURE_MATRIX_FILE)
    logger.info(f"Loading feature matrix from {filepath}")

    # Load data
    adr_cols = [f'adr_{cat}' for cat in config.ADR_CATEGORIES]
    usecols = ['PRIMARYID'] + config.FEATURE_COLS + adr_cols

    dtype_mapping = {col: 'float32' for col in config.FEATURE_COLS}
    for col in adr_cols:
        dtype_mapping[col] = 'float32'

    df = pd.read_csv(filepath, usecols=usecols, dtype=dtype_mapping)
    logger.info(f"Loaded {len(df):,} samples")

    # Prepare features and labels
    X = df[config.FEATURE_COLS].copy()
    if X['age_years'].isna().any():
        X['age_years'] = X['age_years'].fillna(X['age_years'].median())

    y = df[adr_cols].copy()
    has_adr = (y.sum(axis=1) > 0).astype(int)

    # Recreate the same train/test split as model training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=has_adr
    )

    logger.info(f"Test set size: {len(X_test):,} samples")

    # Scale features
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test


def sample_data_for_shap(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    config: SHAPConfig,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample data for SHAP computation (memory optimization).

    Returns:
        X_sample, y_sample
    """
    if len(X_test) <= config.SHAP_SAMPLE_SIZE:
        logger.info(f"Using full test set ({len(X_test):,} samples)")
        return X_test, y_test

    np.random.seed(config.RANDOM_STATE)
    sample_idx = np.random.choice(
        X_test.index,
        size=config.SHAP_SAMPLE_SIZE,
        replace=False
    )

    X_sample = X_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx]

    logger.info(f"Sampled {len(X_sample):,} samples for SHAP computation")

    return X_sample, y_sample


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

class SHAPAnalyzer:
    """Computes and manages SHAP values for multi-output model"""

    def __init__(self, config: SHAPConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.shap_values = {}
        self.explainers = {}
        self.feature_importance = {}

    def compute_shap_values(
        self,
        model: MultiOutputClassifier,
        X_sample: pd.DataFrame,
        X_background: Optional[pd.DataFrame] = None
    ):
        """
        Compute SHAP values for all ADR categories.

        Args:
            model: Trained MultiOutputClassifier
            X_sample: Samples to explain
            X_background: Background samples for explainer
        """
        print("\n" + "="*60)
        print("COMPUTING SHAP VALUES")
        print("="*60)

        for i, category in enumerate(tqdm(self.config.ADR_CATEGORIES, desc="SHAP Analysis")):
            self.logger.info(f"Computing SHAP values for {category}...")

            # Get the individual estimator for this ADR
            estimator = model.estimators_[i]

            try:
                # Try TreeExplainer first (fastest for tree-based models)
                explainer = shap.TreeExplainer(estimator)
                shap_vals = explainer.shap_values(X_sample)

                # For binary classification, shap_values returns list [class0, class1]
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # Use positive class

                self.explainers[category] = explainer
                self.shap_values[category] = shap_vals

                self.logger.info(f"TreeExplainer successful for {category}")

            except Exception as e:
                self.logger.warning(f"TreeExplainer failed for {category}: {e}")
                self.logger.info("Falling back to KernelExplainer...")

                # Fall back to KernelExplainer
                if X_background is None:
                    X_background = shap.sample(X_sample, self.config.SHAP_BACKGROUND_SIZE)

                explainer = shap.KernelExplainer(
                    estimator.predict_proba,
                    X_background
                )
                shap_vals = explainer.shap_values(X_sample)

                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]

                self.explainers[category] = explainer
                self.shap_values[category] = shap_vals

            # Compute feature importance (mean |SHAP|)
            importance = np.abs(shap_vals).mean(axis=0)
            self.feature_importance[category] = pd.DataFrame({
                'feature': self.config.FEATURE_COLS,
                'importance': importance,
                'mean_shap': shap_vals.mean(axis=0)
            }).sort_values('importance', ascending=False)

        self.logger.info("SHAP computation complete for all categories")

    def get_global_importance(self) -> pd.DataFrame:
        """
        Get aggregated global feature importance across all ADR categories.

        Returns:
            DataFrame with overall importance rankings
        """
        all_importance = []

        for category in self.config.ADR_CATEGORIES:
            df = self.feature_importance[category].copy()
            df['category'] = category
            all_importance.append(df)

        combined = pd.concat(all_importance, ignore_index=True)

        # Aggregate across categories
        overall = combined.groupby('feature').agg({
            'importance': 'mean',
            'mean_shap': 'mean'
        }).reset_index()

        overall = overall.sort_values('importance', ascending=False)
        overall['rank'] = range(1, len(overall) + 1)

        return overall

    def identify_high_risk_cases(
        self,
        model: MultiOutputClassifier,
        X_sample: pd.DataFrame,
        category: str,
        n_cases: int = 3
    ) -> List[int]:
        """
        Identify high-risk predictions for waterfall plots.

        Args:
            model: Trained model
            X_sample: Sample data
            category: ADR category
            n_cases: Number of cases to select

        Returns:
            List of indices for high-risk cases
        """
        cat_idx = self.config.ADR_CATEGORIES.index(category)
        estimator = model.estimators_[cat_idx]

        # Get prediction probabilities
        proba = estimator.predict_proba(X_sample)[:, 1]

        # Select top N highest probability cases
        top_indices = np.argsort(proba)[::-1][:n_cases]

        return [X_sample.index[i] for i in top_indices]


# =============================================================================
# VISUALIZATION
# =============================================================================

class SHAPPlotGenerator:
    """Generates SHAP visualizations"""

    def __init__(self, config: SHAPConfig, analyzer: SHAPAnalyzer):
        self.config = config
        self.analyzer = analyzer
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib for publication quality"""
        plt.rcParams.update({
            'figure.dpi': PlotConfig.DPI,
            'savefig.dpi': PlotConfig.DPI,
            'savefig.bbox': 'tight',
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11
        })

    def _get_display_names(self, feature_names: List[str]) -> List[str]:
        """Convert feature names to display-friendly names"""
        return [self.config.FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names]

    def plot_summary_all_categories(
        self,
        X_sample: pd.DataFrame,
        output_dir: str
    ):
        """Generate summary plots for all ADR categories in a grid"""
        fig, axes = plt.subplots(2, 3, figsize=PlotConfig.FIGSIZE_LARGE)
        axes = axes.flatten()

        for i, category in enumerate(self.config.ADR_CATEGORIES):
            plt.sca(axes[i])

            shap_vals = self.analyzer.shap_values[category]

            # Create display-friendly feature names
            X_display = X_sample.copy()
            X_display.columns = self._get_display_names(X_sample.columns.tolist())

            shap.summary_plot(
                shap_vals,
                X_display,
                plot_type='bar',
                show=False,
                max_display=10
            )

            axes[i].set_title(self.config.ADR_DISPLAY_NAMES[category], fontsize=11)

        # Hide the 6th subplot
        axes[5].axis('off')

        plt.suptitle('SHAP Feature Importance by ADR Category', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, 'shap_summary_all_categories.png')
        plt.savefig(save_path, dpi=PlotConfig.DPI, bbox_inches='tight')
        plt.close()

    def plot_summary_single(
        self,
        category: str,
        X_sample: pd.DataFrame,
        output_dir: str,
        plot_type: str = 'beeswarm'
    ):
        """Generate detailed summary plot for single ADR category"""
        plt.figure(figsize=PlotConfig.FIGSIZE_SMALL)

        shap_vals = self.analyzer.shap_values[category]

        X_display = X_sample.copy()
        X_display.columns = self._get_display_names(X_sample.columns.tolist())

        if plot_type == 'beeswarm':
            shap.summary_plot(
                shap_vals,
                X_display,
                plot_type='dot',
                show=False,
                max_display=15
            )
        else:
            shap.summary_plot(
                shap_vals,
                X_display,
                plot_type='bar',
                show=False,
                max_display=15
            )

        plt.title(f'SHAP Analysis - {self.config.ADR_DISPLAY_NAMES[category]}',
                  fontsize=12, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'shap_summary_{category}.png')
        plt.savefig(save_path, dpi=PlotConfig.DPI, bbox_inches='tight')
        plt.close()

    def plot_waterfall_top3(
        self,
        category: str,
        model: MultiOutputClassifier,
        X_sample: pd.DataFrame,
        output_dir: str
    ):
        """Generate waterfall plots for top 3 high-risk cases"""
        # Get high-risk case indices
        high_risk_indices = self.analyzer.identify_high_risk_cases(
            model, X_sample, category, n_cases=self.config.N_HIGH_RISK_CASES
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        shap_vals = self.analyzer.shap_values[category]
        explainer = self.analyzer.explainers[category]

        for idx, (case_idx, ax) in enumerate(zip(high_risk_indices, axes)):
            plt.sca(ax)

            # Get the position of this case in the sample
            case_pos = X_sample.index.get_loc(case_idx)

            # Create SHAP Explanation object
            feature_names = self._get_display_names(self.config.FEATURE_COLS)

            # Get expected value
            if hasattr(explainer, 'expected_value'):
                expected_val = explainer.expected_value
                if isinstance(expected_val, (list, np.ndarray)):
                    expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]
            else:
                expected_val = 0

            explanation = shap.Explanation(
                values=shap_vals[case_pos],
                base_values=expected_val,
                data=X_sample.iloc[case_pos].values,
                feature_names=feature_names
            )

            shap.waterfall_plot(explanation, show=False, max_display=10)
            ax.set_title(f'Case {idx+1} (High Risk)', fontsize=10)

        plt.suptitle(
            f'High-Risk Case Explanations - {self.config.ADR_DISPLAY_NAMES[category]}',
            fontsize=12, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'waterfall_{category}_top3.png')
        plt.savefig(save_path, dpi=PlotConfig.DPI, bbox_inches='tight')
        plt.close()

    def plot_dependence_top5(
        self,
        category: str,
        X_sample: pd.DataFrame,
        output_dir: str
    ):
        """Generate dependence plots for top 5 most important features"""
        # Get top 5 features for this category
        top_features = self.analyzer.feature_importance[category].head(
            self.config.N_TOP_FEATURES
        )['feature'].tolist()

        fig, axes = plt.subplots(2, 3, figsize=PlotConfig.FIGSIZE_LARGE)
        axes = axes.flatten()

        shap_vals = self.analyzer.shap_values[category]

        for i, feature in enumerate(top_features):
            plt.sca(axes[i])

            feature_idx = self.config.FEATURE_COLS.index(feature)
            display_name = self.config.FEATURE_DISPLAY_NAMES.get(feature, feature)

            shap.dependence_plot(
                feature_idx,
                shap_vals,
                X_sample,
                feature_names=self.config.FEATURE_COLS,
                show=False,
                ax=axes[i]
            )

            axes[i].set_xlabel(display_name)
            axes[i].set_title(f'{display_name}', fontsize=10)

        # Hide 6th subplot, add summary
        axes[5].axis('off')
        axes[5].text(
            0.5, 0.5,
            f"Top 5 Features for\n{self.config.ADR_DISPLAY_NAMES[category]}",
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            transform=axes[5].transAxes
        )

        plt.suptitle(
            f'Feature Dependence Plots - {self.config.ADR_DISPLAY_NAMES[category]}',
            fontsize=12, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'dependence_{category}_top5.png')
        plt.savefig(save_path, dpi=PlotConfig.DPI, bbox_inches='tight')
        plt.close()

    def plot_feature_importance_bar(
        self,
        category: str,
        output_dir: str
    ):
        """Generate bar chart of feature importance for single category"""
        plt.figure(figsize=PlotConfig.FIGSIZE_SMALL)

        importance_df = self.analyzer.feature_importance[category].head(15).copy()
        importance_df['display_name'] = importance_df['feature'].map(
            self.config.FEATURE_DISPLAY_NAMES
        )

        colors = ['#d62728' if x > 0 else '#1f77b4'
                  for x in importance_df['mean_shap']]

        plt.barh(
            range(len(importance_df)),
            importance_df['importance'],
            color=colors
        )

        plt.yticks(range(len(importance_df)), importance_df['display_name'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Feature Importance - {self.config.ADR_DISPLAY_NAMES[category]}',
                  fontweight='bold')
        plt.gca().invert_yaxis()

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', label='Increases Risk'),
            Patch(facecolor='#1f77b4', label='Decreases Risk')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        save_path = os.path.join(output_dir, f'feature_importance_{category}.png')
        plt.savefig(save_path, dpi=PlotConfig.DPI, bbox_inches='tight')
        plt.close()


# =============================================================================
# CLINICAL REPORT GENERATION
# =============================================================================

class ClinicalReportGenerator:
    """Generates clinical interpretation reports"""

    def __init__(self, config: SHAPConfig, analyzer: SHAPAnalyzer):
        self.config = config
        self.analyzer = analyzer

    def identify_risk_factors(self, category: str, n_factors: int = 10) -> pd.DataFrame:
        """Identify top risk factors (positive SHAP values)"""
        importance_df = self.analyzer.feature_importance[category].copy()

        # Filter for positive SHAP (increases risk)
        risk_factors = importance_df[importance_df['mean_shap'] > 0].head(n_factors)

        risk_factors['display_name'] = risk_factors['feature'].map(
            self.config.FEATURE_DISPLAY_NAMES
        )
        risk_factors['direction'] = 'Increases Risk'
        risk_factors['rank'] = range(1, len(risk_factors) + 1)

        return risk_factors[['rank', 'display_name', 'importance', 'mean_shap', 'direction']]

    def identify_protective_factors(self, category: str, n_factors: int = 5) -> pd.DataFrame:
        """Identify top protective factors (negative SHAP values)"""
        importance_df = self.analyzer.feature_importance[category].copy()

        # Filter for negative SHAP (decreases risk)
        protective = importance_df[importance_df['mean_shap'] < 0].copy()
        protective = protective.sort_values('mean_shap').head(n_factors)

        protective['display_name'] = protective['feature'].map(
            self.config.FEATURE_DISPLAY_NAMES
        )
        protective['direction'] = 'Decreases Risk'
        protective['rank'] = range(1, len(protective) + 1)

        return protective[['rank', 'display_name', 'importance', 'mean_shap', 'direction']]

    def generate_full_report(self, output_dir: str):
        """Generate comprehensive clinical interpretation report"""
        report_path = os.path.join(output_dir, 'clinical_interpretation_report.md')

        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Clinical Interpretation Report: Hepatitis B ADR Prediction\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report provides SHAP-based explanations for the Hepatitis B ")
            f.write("Adverse Drug Reaction (ADR) prediction model. SHAP (SHapley Additive ")
            f.write("exPlanations) values quantify each feature's contribution to predictions, ")
            f.write("enabling clinical interpretation of risk factors.\n\n")

            # Overall top risk factors
            f.write("## Overall Top 10 Risk Factors\n\n")
            overall_importance = self.analyzer.get_global_importance()
            top_risk = overall_importance[overall_importance['mean_shap'] > 0].head(10)

            f.write("| Rank | Feature | Mean Importance | Direction |\n")
            f.write("|------|---------|-----------------|----------|\n")
            for i, row in top_risk.iterrows():
                display_name = self.config.FEATURE_DISPLAY_NAMES.get(row['feature'], row['feature'])
                f.write(f"| {row['rank']} | {display_name} | {row['importance']:.4f} | Increases Risk |\n")
            f.write("\n")

            # Overall protective factors
            f.write("## Top 5 Protective Factors\n\n")
            protective = overall_importance[overall_importance['mean_shap'] < 0].head(5)

            f.write("| Rank | Feature | Mean Importance | Direction |\n")
            f.write("|------|---------|-----------------|----------|\n")
            for idx, (i, row) in enumerate(protective.iterrows(), 1):
                display_name = self.config.FEATURE_DISPLAY_NAMES.get(row['feature'], row['feature'])
                f.write(f"| {idx} | {display_name} | {row['importance']:.4f} | Decreases Risk |\n")
            f.write("\n")

            # Per-ADR Analysis
            f.write("---\n\n")
            f.write("## ADR-Specific Analysis\n\n")

            for category in self.config.ADR_CATEGORIES:
                display_name = self.config.ADR_DISPLAY_NAMES[category]
                f.write(f"### {display_name}\n\n")

                # Risk factors
                risk_factors = self.identify_risk_factors(category, n_factors=5)
                f.write("**Top 5 Risk Factors:**\n\n")
                f.write("| Rank | Feature | SHAP Importance |\n")
                f.write("|------|---------|----------------|\n")
                for _, row in risk_factors.iterrows():
                    f.write(f"| {row['rank']} | {row['display_name']} | {row['importance']:.4f} |\n")
                f.write("\n")

                # Protective factors
                protective = self.identify_protective_factors(category, n_factors=3)
                if len(protective) > 0:
                    f.write("**Protective Factors:**\n\n")
                    f.write("| Rank | Feature | SHAP Importance |\n")
                    f.write("|------|---------|----------------|\n")
                    for _, row in protective.iterrows():
                        f.write(f"| {row['rank']} | {row['display_name']} | {row['importance']:.4f} |\n")
                    f.write("\n")

                f.write("\n")

            # Methodology
            f.write("---\n\n")
            f.write("## Methodology\n\n")
            f.write("- **SHAP Algorithm:** TreeExplainer for tree-based ensemble models\n")
            f.write(f"- **Sample Size:** {self.config.SHAP_SAMPLE_SIZE:,} patients from test set\n")
            f.write("- **Interpretation:** Positive SHAP values indicate features that increase ")
            f.write("ADR risk; negative values indicate protective factors\n")
            f.write("- **Feature Importance:** Calculated as mean absolute SHAP value across all samples\n\n")

            # Limitations
            f.write("## Limitations\n\n")
            f.write("1. SHAP values represent average feature contributions and may vary for individual patients\n")
            f.write("2. Correlations between features may affect individual SHAP value interpretations\n")
            f.write("3. This analysis is based on observational FAERS data and should not be used for ")
            f.write("individual clinical decisions without additional validation\n")
            f.write("4. Drug-specific risks require consideration of indication, dose, and duration\n\n")

            f.write("---\n\n")
            f.write("*Report generated by SHAP Explainability Pipeline*\n")

        # Also save as plain text
        txt_path = os.path.join(output_dir, 'clinical_interpretation_report.txt')
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Simple markdown to text conversion
        content = content.replace('# ', '').replace('## ', '').replace('### ', '')
        content = content.replace('**', '').replace('---', '-'*60)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def save_factor_tables(self, output_dir: str):
        """Save risk and protective factor tables as CSV"""
        # All risk factors
        all_risk = []
        for category in self.config.ADR_CATEGORIES:
            risk = self.identify_risk_factors(category, n_factors=10)
            risk['adr_category'] = category
            all_risk.append(risk)

        risk_df = pd.concat(all_risk, ignore_index=True)
        risk_path = os.path.join(output_dir, 'risk_factors_all_categories.csv')
        risk_df.to_csv(risk_path, index=False)

        # All protective factors
        all_protective = []
        for category in self.config.ADR_CATEGORIES:
            protective = self.identify_protective_factors(category, n_factors=5)
            protective['adr_category'] = category
            all_protective.append(protective)

        protective_df = pd.concat(all_protective, ignore_index=True)
        protective_path = os.path.join(output_dir, 'protective_factors_all_categories.csv')
        protective_df.to_csv(protective_path, index=False)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_shap_analysis_pipeline():
    """Main orchestration function for SHAP analysis"""
    config = SHAPConfig()

    # Setup
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(config.OUTPUT_DIR)

    print("\n" + "="*80)
    print("HEPATITIS B ADR PREDICTION - SHAP EXPLAINABILITY ANALYSIS")
    print("="*80)
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print("="*80 + "\n")

    logger.info("Starting SHAP analysis pipeline")

    try:
        # 1. Load model and scaler
        model, scaler = load_model_and_scaler(config, logger)

        # 2. Load and prepare data
        X_train, X_test, y_train, y_test = load_and_prepare_data(config, scaler, logger)

        # 3. Sample data for SHAP computation
        X_sample, y_sample = sample_data_for_shap(X_test, y_test, config, logger)

        # 4. Create SHAP analyzer
        analyzer = SHAPAnalyzer(config, logger)

        # 5. Compute SHAP values
        analyzer.compute_shap_values(model, X_sample)

        # 6. Create plot generator
        plot_generator = SHAPPlotGenerator(config, analyzer)

        # 7. Generate visualizations
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        print("Generating summary plots...")
        plot_generator.plot_summary_all_categories(X_sample, config.OUTPUT_DIR)

        for category in tqdm(config.ADR_CATEGORIES, desc="Individual Summaries"):
            plot_generator.plot_summary_single(category, X_sample, config.OUTPUT_DIR, 'beeswarm')
            plot_generator.plot_feature_importance_bar(category, config.OUTPUT_DIR)

        print("Generating waterfall plots...")
        for category in tqdm(config.ADR_CATEGORIES, desc="Waterfall Plots"):
            try:
                plot_generator.plot_waterfall_top3(category, model, X_sample, config.OUTPUT_DIR)
            except Exception as e:
                logger.warning(f"Waterfall plot failed for {category}: {e}")

        print("Generating dependence plots...")
        for category in tqdm(config.ADR_CATEGORIES, desc="Dependence Plots"):
            try:
                plot_generator.plot_dependence_top5(category, X_sample, config.OUTPUT_DIR)
            except Exception as e:
                logger.warning(f"Dependence plot failed for {category}: {e}")

        # 8. Generate clinical report
        print("\n" + "="*60)
        print("GENERATING CLINICAL REPORT")
        print("="*60)

        report_generator = ClinicalReportGenerator(config, analyzer)
        report_generator.generate_full_report(config.OUTPUT_DIR)
        report_generator.save_factor_tables(config.OUTPUT_DIR)

        logger.info("Clinical report generated successfully")

        # Print summary
        print("\n" + "="*80)
        print("SHAP ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
        print("\nGenerated files:")
        print("  - shap_summary_all_categories.png")
        print("  - shap_summary_{category}.png (x5)")
        print("  - waterfall_{category}_top3.png (x5)")
        print("  - dependence_{category}_top5.png (x5)")
        print("  - feature_importance_{category}.png (x5)")
        print("  - clinical_interpretation_report.md")
        print("  - risk_factors_all_categories.csv")
        print("  - protective_factors_all_categories.csv")
        print("="*80 + "\n")

        logger.info("Pipeline completed successfully")

        return analyzer

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_shap_analysis_pipeline()
