# phase2_visualizations.py

"""
Phase 2 Visualizations
Generate plots and figures for cohort comparison and signal detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

class VizConfig:
    DATA_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase2_analysis"
    OUTPUT_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase2_visualizations"

config = VizConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_phase2_data():
    """Load Phase 2 analysis results"""
    print("\n" + "="*80)
    print("LOADING PHASE 2 DATA FOR VISUALIZATION")
    print("="*80)
    
    files = {
        'cohorts': 'cohort_assignments.csv',
        'pairs': 'drug_adr_pairs.csv',
        'signals': 'all_signals.csv',
        'comparison': 'cohort_comparison.csv',
        'severity': 'severity_by_cohort.csv'
    }
    
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
            print(f"✓ Loaded {key}")
        else:
            print(f"✗ Missing {key}")
            data[key] = None
    
    return data

# ============================================================================
# COHORT DISTRIBUTION PLOTS
# ============================================================================

def plot_cohort_distribution(data):
    """Plot cohort sizes and distributions"""
    cohorts_df = data['cohorts']
    
    if cohorts_df is None:
        return
    
    cohort_counts = cohorts_df['cohort'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    bars = ax.bar(range(len(cohort_counts)), cohort_counts.values, color=colors)
    
    ax.set_xticks(range(len(cohort_counts)))
    ax.set_xticklabels(cohort_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Hepatitis B Patient Cohort Distribution', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'cohort_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: cohort_distribution.png")

# ============================================================================
# SEVERITY COMPARISON
# ============================================================================

def plot_severity_comparison(data):
    """Compare severity distributions across cohorts"""
    severity_df = data['severity']
    
    if severity_df is None:
        return
    
    # Focus on main cohorts
    main_cohorts = ['HBV_MONO', 'HIV_HBV_COINFECTION']
    severity_df = severity_df[severity_df['cohort'].isin(main_cohorts)]
    
    # Prepare data for stacked bar chart
    severity_cols = ['severity_1_count', 'severity_2_count', 'severity_3_count', 
                     'severity_4_count', 'severity_5_count']
    severity_labels = ['Grade 1\n(Other)', 'Grade 2\n(RI/CA)', 'Grade 3\n(Hosp/Dis)', 
                       'Grade 4\n(Life-threat)', 'Grade 5\n(Death)']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(severity_df))
    width = 0.6
    
    colors = ['#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    bottom = np.zeros(len(severity_df))
    
    for idx, (col, label) in enumerate(zip(severity_cols, severity_labels)):
        values = severity_df[col].values
        ax.bar(x, values, width, label=label, bottom=bottom, color=colors[idx])
        bottom += values
    
    ax.set_ylabel('Number of ADR Records', fontsize=12, fontweight='bold')
    ax.set_title('ADR Severity Distribution by Cohort', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(severity_df['cohort'].values)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'severity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: severity_comparison.png")

# ============================================================================
# TOP SIGNALS VISUALIZATION
# ============================================================================

def plot_top_signals(data):
    """Visualize top detected signals"""
    signals_df = data['signals']
    
    if signals_df is None or len(signals_df) == 0:
        return
    
    # Get top 20 signals by ROR
    top_signals = signals_df[signals_df['is_signal']].nlargest(20, 'ror')
    
    if len(top_signals) == 0:
        print("No significant signals detected")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create labels
    labels = [f"{row['drug'][:15]}\n{row['adr'][:40]}" for _, row in top_signals.iterrows()]
    
    y_pos = np.arange(len(labels))
    
    # Plot ROR with error bars
    ror = top_signals['ror'].values
    ror_lower = top_signals['ror_lower_ci'].values
    ror_upper = top_signals['ror_upper_ci'].values
    
    # Calculate error bar values
    err_lower = ror - ror_lower
    err_upper = ror_upper - ror
    
    # Color by cohort
    colors_map = {
        'HBV_MONO': '#2ecc71',
        'HIV_HBV_COINFECTION': '#e74c3c',
        'HBV_AUTOIMMUNE': '#f39c12'
    }
    colors = [colors_map.get(row['cohort'], '#95a5a6') for _, row in top_signals.iterrows()]
    
    ax.barh(y_pos, ror, xerr=[err_lower, err_upper], 
            color=colors, alpha=0.8, capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Reporting Odds Ratio (ROR) with 95% CI', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Drug-ADR Signal Detections', fontsize=14, fontweight='bold')
    ax.axvline(x=2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Signal Threshold (ROR=2)')
    
    # Legend for cohorts
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='HBV Mono-infection'),
        Patch(facecolor='#e74c3c', label='HIV/HBV Co-infection'),
        Patch(facecolor='#f39c12', label='HBV + Autoimmune')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'top_signals.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: top_signals.png")

# ============================================================================
# COHORT COMPARISON HEATMAP
# ============================================================================

def plot_cohort_comparison_heatmap(data):
    """Heatmap comparing ADR frequencies between cohorts"""
    comparison_df = data['comparison']
    
    if comparison_df is None:
        return
    
    # Get top 30 ADRs with largest differences
    comparison_df['abs_diff'] = np.abs(np.log2(comparison_df['risk_ratio']))
    top_adrs = comparison_df.nlargest(30, 'abs_diff')
    
    # Prepare data for heatmap
    heatmap_data = top_adrs[['adr', 'hbv_mono_prop', 'hiv_hbv_prop']].set_index('adr')
    heatmap_data.columns = ['HBV Mono', 'HIV/HBV']
    
    # Convert to percentages
    heatmap_data = heatmap_data * 100
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'ADR Frequency (%)'}, ax=ax)
    
    ax.set_title('Top 30 ADRs: Frequency Comparison Between Cohorts', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Adverse Drug Reaction', fontsize=12, fontweight='bold')
    ax.set_xlabel('Cohort', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'cohort_comparison_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: cohort_comparison_heatmap.png")

# ============================================================================
# VOLCANO PLOT FOR COHORT COMPARISON
# ============================================================================

def plot_volcano(data):
    """Volcano plot showing ADR differences between cohorts"""
    comparison_df = data['comparison']
    
    if comparison_df is None:
        return
    
    # Calculate log2 risk ratio and -log10 p-value
    comparison_df['log2_rr'] = np.log2(comparison_df['risk_ratio'])
    comparison_df['neg_log10_p'] = -np.log10(comparison_df['p_value'])
    
    # Remove infinite and NaN values
    plot_df = comparison_df[
        np.isfinite(comparison_df['log2_rr']) & 
        np.isfinite(comparison_df['neg_log10_p'])
    ].copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Significance thresholds
    p_threshold = 0.05
    rr_threshold = 2  # 2-fold difference
    
    # Color code points
    colors = []
    for _, row in plot_df.iterrows():
        if row['p_value'] < p_threshold:
            if row['log2_rr'] > np.log2(rr_threshold):
                colors.append('#e74c3c')  # Enriched in HBV Mono
            elif row['log2_rr'] < -np.log2(rr_threshold):
                colors.append('#3498db')  # Enriched in HIV/HBV
            else:
                colors.append('#95a5a6')  # Significant but small effect
        else:
            colors.append('#bdc3c7')  # Not significant
    
    scatter = ax.scatter(plot_df['log2_rr'], plot_df['neg_log10_p'], 
                        c=colors, alpha=0.6, s=30)
    
    # Add threshold lines
    ax.axhline(y=-np.log10(p_threshold), color='black', linestyle='--', 
               linewidth=1, alpha=0.5, label=f'p = {p_threshold}')
    ax.axvline(x=np.log2(rr_threshold), color='black', linestyle='--', 
               linewidth=1, alpha=0.5)
    ax.axvline(x=-np.log2(rr_threshold), color='black', linestyle='--', 
               linewidth=1, alpha=0.5)
    
    # Label top significant ADRs
    top_enriched_hbv = plot_df[
        (plot_df['log2_rr'] > np.log2(rr_threshold)) & 
        (plot_df['p_value'] < p_threshold)
    ].nlargest(5, 'neg_log10_p')
    
    top_enriched_hiv = plot_df[
        (plot_df['log2_rr'] < -np.log2(rr_threshold)) & 
        (plot_df['p_value'] < p_threshold)
    ].nlargest(5, 'neg_log10_p')
    
    for _, row in top_enriched_hbv.iterrows():
        ax.annotate(row['adr'][:20], 
                   (row['log2_rr'], row['neg_log10_p']),
                   fontsize=8, alpha=0.8)
    
    for _, row in top_enriched_hiv.iterrows():
        ax.annotate(row['adr'][:20], 
                   (row['log2_rr'], row['neg_log10_p']),
                   fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Log₂ Risk Ratio (HBV Mono / HIV-HBV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('-Log₁₀ p-value', fontsize=12, fontweight='bold')
    ax.set_title('Volcano Plot: ADR Differences Between Cohorts', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Enriched in HBV Mono'),
        Patch(facecolor='#3498db', label='Enriched in HIV/HBV'),
        Patch(facecolor='#95a5a6', label='Significant (small effect)'),
        Patch(facecolor='#bdc3c7', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'volcano_plot.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: volcano_plot.png")

# ============================================================================
# DRUG CLASS ADR FREQUENCY
# ============================================================================

def plot_drug_adr_frequency(data):
    """Plot ADR frequencies for each HBV antiviral drug class"""
    pairs_df = data['pairs']
    
    if pairs_df is None:
        return
    
    # Focus on HBV_MONO cohort
    mono_pairs = pairs_df[pairs_df['cohort'] == 'HBV_MONO']
    
    # Get top 10 ADRs for each drug
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    drugs = mono_pairs['drug_class'].unique()
    
    for idx, drug in enumerate(drugs):
        if idx >= len(axes):
            break
        
        drug_pairs = mono_pairs[mono_pairs['drug_class'] == drug]
        top_adrs = drug_pairs['adr'].value_counts().head(10)
        
        axes[idx].barh(range(len(top_adrs)), top_adrs.values, color='#3498db')
        axes[idx].set_yticks(range(len(top_adrs)))
        axes[idx].set_yticklabels([adr[:30] for adr in top_adrs.index], fontsize=9)
        axes[idx].set_xlabel('Frequency', fontsize=10, fontweight='bold')
        axes[idx].set_title(f'{drug}', fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
    
    # Hide unused subplots
    for idx in range(len(drugs), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Top 10 ADRs by HBV Antiviral Drug (HBV Mono-infection)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'drug_adr_frequency.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: drug_adr_frequency.png")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_visualizations():
    """Generate all visualizations"""
    print("\n" + "🎨"*40)
    print("GENERATING PHASE 2 VISUALIZATIONS")
    print("🎨"*40)
    
    # Load data
    data = load_phase2_data()
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_cohort_distribution(data)
    plot_severity_comparison(data)
    plot_top_signals(data)
    plot_cohort_comparison_heatmap(data)
    plot_volcano(data)
    plot_drug_adr_frequency(data)
    
    print("\n" + "✅"*40)
    print(f"ALL VISUALIZATIONS SAVED TO: {config.OUTPUT_DIR}")
    print("✅"*40)

if __name__ == "__main__":
    generate_all_visualizations()