# phase3_feature_extraction.py

"""
Phase 3A: Feature Extraction from FAERS Data
Extract realistic, available features for ML model training

Features extracted:
- Demographics: Age, Sex, Geographic region
- Severity: Outcome codes, severe disease indicator
- Polypharmacy: Drug count per patient
- Cohort: HBV_MONO, HIV_HBV, AUTOIMMUNE
- Reporter type: MD/HP, Lawyer, Consumer
- Drug risk scores from Phase 2 signals
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Phase3Config:
    """Configuration for Phase 3 feature extraction"""
    
    # Input paths (Phase 2 outputs)
    DATA_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase2_analysis"
    PHASE1_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\hepatitis_b_data"
    
    # Output paths
    OUTPUT_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase3_features"
    
    # ADR categories for multi-label classification
    ADR_CATEGORIES = {
        'hepatotoxicity': [
            'Hepatic failure', 'Hepatotoxicity', 'Liver disorder', 
            'Hepatic cirrhosis', 'Hepatitis', 'Hepatic necrosis',
            'Hepatomegaly', 'Jaundice', 'Liver function test abnormal',
            'Hepatocellular carcinoma', 'Hepatic fibrosis'
        ],
        'nephrotoxicity': [
            'Renal failure', 'Renal impairment', 'Chronic kidney disease',
            'Renal injury', 'Acute kidney injury', 'Nephropathy',
            'Renal tubular disorder', 'Fanconi syndrome acquired',
            'Blood creatinine increased', 'Proteinuria'
        ],
        'bone_tooth': [
            'Bone density decreased', 'Osteoporosis', 'Osteopenia',
            'Tooth loss', 'Osteonecrosis', 'Bone loss', 'Skeletal injury',
            'Osteomalacia', 'Bone fracture', 'Dental caries',
            'Periodontal disease', 'Tooth disorder'
        ],
        'hematologic': [
            'Neutropenia', 'Anaemia', 'Thrombocytopenia', 'Leukopenia',
            'Pancytopenia', 'White blood cell count decreased',
            'Platelet count decreased', 'Haemoglobin decreased',
            'Bone marrow failure', 'Lymphopenia'
        ],
        'neurologic': [
            'Neuropathy peripheral', 'Headache', 'Dizziness', 'Seizure',
            'Encephalopathy', 'Tremor', 'Paraesthesia', 'Cognitive disorder',
            'Confusion', 'Somnolence', 'Insomnia', 'Peripheral sensory neuropathy'
        ]
    }
    
    # Geographic regions
    REGION_MAPPING = {
        'US': 'North_America',
        'CA': 'North_America',
        'MX': 'North_America',
        'GB': 'Europe',
        'DE': 'Europe',
        'FR': 'Europe',
        'IT': 'Europe',
        'ES': 'Europe',
        'JP': 'Asia',
        'CN': 'Asia',
        'KR': 'Asia',
        'TW': 'Asia',
        'IN': 'Asia'
    }

config = Phase3Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_phase2_data():
    """Load Phase 2 outputs and Phase 1 raw data"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    data = {}
    
    # Phase 2 outputs
    print("\nPhase 2 outputs:")
    phase2_files = {
        'pairs': 'drug_adr_pairs.csv',
        'signals': 'all_signals.csv',
        'cohorts': 'cohort_assignments.csv'
    }
    
    for key, filename in phase2_files.items():
        filepath = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
            print(f"  ✓ Loaded {key}: {len(data[key]):,} records")
        else:
            print(f"  ✗ Missing {key}: {filepath}")
            return None
    
    # Phase 1 raw data (for demographics)
    print("\nPhase 1 raw data:")
    phase1_files = {
        'demo': 'hepb_demo.csv',
        'drug': 'hepb_drug.csv',
        'outc': 'hepb_outc.csv'
    }
    
    for key, filename in phase1_files.items():
        filepath = os.path.join(config.PHASE1_DIR, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath, dtype=str)
            print(f"  ✓ Loaded {key}: {len(data[key]):,} records")
        else:
            print(f"  ✗ Missing {key}: {filepath}")
            return None
    
    return data

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_demographics(demo_df):
    """
    Extract demographic features:
    - Age categories
    - Sex
    - Geographic region
    """
    print("\n" + "="*80)
    print("EXTRACTING DEMOGRAPHICS")
    print("="*80)
    
    demo_features = demo_df[['PRIMARYID']].copy()
    
    # Age categories
    demo_df['AGE_YEARS'] = pd.to_numeric(demo_df['AGE_YEARS'], errors='coerce')
    
    def categorize_age(age):
        if pd.isna(age):
            return 'Unknown'
        elif age < 18:
            return '<18'
        elif age < 41:
            return '18-40'
        elif age < 66:
            return '41-65'
        else:
            return '>65'
    
    demo_features['age_category'] = demo_df['AGE_YEARS'].apply(categorize_age)
    demo_features['age_years'] = demo_df['AGE_YEARS']  # Keep continuous for modeling
    
    # Sex
    demo_features['sex'] = demo_df['SEX'].fillna('Unknown')
    demo_features['sex_binary'] = (demo_features['sex'] == 'M').astype(int)
    demo_features['sex_female'] = (demo_features['sex'] == 'F').astype(int)
    
    # Geographic region
    demo_features['country'] = demo_df['REPORTER_COUNTRY'].fillna('Unknown')
    demo_features['region'] = demo_features['country'].map(
        lambda x: config.REGION_MAPPING.get(x, 'Other')
    )
    
    # Reporter type
    demo_features['reporter_type'] = demo_df['OCCP_COD'].fillna('Unknown')
    demo_features['reporter_md'] = demo_features['reporter_type'].isin(['MD', 'HP']).astype(int)
    demo_features['reporter_lawyer'] = (demo_features['reporter_type'] == 'LW').astype(int)
    demo_features['reporter_consumer'] = (demo_features['reporter_type'] == 'CN').astype(int)
    
    print(f"  Extracted demographics for {len(demo_features):,} patients")
    print(f"\n  Age distribution:")
    print(demo_features['age_category'].value_counts())
    print(f"\n  Sex distribution:")
    print(demo_features['sex'].value_counts())
    print(f"\n  Region distribution:")
    print(demo_features['region'].value_counts())
    
    return demo_features


def extract_severity_indicators(outc_df):
    """
    Extract disease severity indicators from outcome codes
    DE = Death, LT = Life-threatening, HO = Hospitalization, etc.
    """
    print("\n" + "="*80)
    print("EXTRACTING SEVERITY INDICATORS")
    print("="*80)
    
    # Get all outcomes per patient
    patient_outcomes = outc_df.groupby('PRIMARYID')['OUTC_COD'].apply(list).reset_index()
    patient_outcomes.columns = ['PRIMARYID', 'outcome_codes']
    
    severity_features = patient_outcomes[['PRIMARYID']].copy()
    
    # Binary indicators for each outcome type
    severity_features['outcome_death'] = patient_outcomes['outcome_codes'].apply(
        lambda x: 1 if 'DE' in x else 0
    )
    severity_features['outcome_life_threat'] = patient_outcomes['outcome_codes'].apply(
        lambda x: 1 if 'LT' in x else 0
    )
    severity_features['outcome_hospitalization'] = patient_outcomes['outcome_codes'].apply(
        lambda x: 1 if 'HO' in x else 0
    )
    severity_features['outcome_disability'] = patient_outcomes['outcome_codes'].apply(
        lambda x: 1 if 'DS' in x else 0
    )
    
    # Severe disease indicator (DE or LT)
    severity_features['severe_disease'] = (
        (severity_features['outcome_death'] == 1) | 
        (severity_features['outcome_life_threat'] == 1)
    ).astype(int)
    
    # Severity score (0-5)
    def calculate_severity_score(codes):
        score = 0
        if 'DE' in codes: score = 5
        elif 'LT' in codes: score = 4
        elif 'HO' in codes or 'DS' in codes: score = 3
        elif 'CA' in codes or 'RI' in codes: score = 2
        else: score = 1
        return score
    
    severity_features['severity_score'] = patient_outcomes['outcome_codes'].apply(
        calculate_severity_score
    )
    
    print(f"  Extracted severity for {len(severity_features):,} patients")
    print(f"\n  Severe disease rate: {severity_features['severe_disease'].mean():.1%}")
    print(f"  Death rate: {severity_features['outcome_death'].mean():.1%}")
    print(f"  Hospitalization rate: {severity_features['outcome_hospitalization'].mean():.1%}")
    
    return severity_features


def extract_polypharmacy(drug_df):
    """
    Extract polypharmacy indicators:
    - Number of concurrent drugs
    - Polypharmacy categories
    """
    print("\n" + "="*80)
    print("EXTRACTING POLYPHARMACY INDICATORS")
    print("="*80)
    
    # Count unique drugs per patient
    drug_counts = drug_df.groupby('PRIMARYID')['DRUGNAME'].nunique().reset_index()
    drug_counts.columns = ['PRIMARYID', 'drug_count']
    
    poly_features = drug_counts.copy()
    
    # Polypharmacy categories
    def categorize_polypharmacy(count):
        if count <= 2:
            return '1-2_drugs'
        elif count <= 5:
            return '3-5_drugs'
        else:
            return '6+_drugs'
    
    poly_features['polypharmacy_category'] = poly_features['drug_count'].apply(
        categorize_polypharmacy
    )
    
    # Binary indicators
    poly_features['polypharmacy_low'] = (poly_features['drug_count'] <= 2).astype(int)
    poly_features['polypharmacy_medium'] = (
        (poly_features['drug_count'] > 2) & (poly_features['drug_count'] <= 5)
    ).astype(int)
    poly_features['polypharmacy_high'] = (poly_features['drug_count'] > 5).astype(int)
    
    print(f"  Extracted polypharmacy for {len(poly_features):,} patients")
    print(f"\n  Drug count distribution:")
    print(poly_features['polypharmacy_category'].value_counts())
    print(f"\n  Mean drugs per patient: {poly_features['drug_count'].mean():.1f}")
    print(f"  Median: {poly_features['drug_count'].median():.0f}")
    
    return poly_features


def extract_cohort_features(cohort_df):
    """
    Extract cohort assignment features
    """
    print("\n" + "="*80)
    print("EXTRACTING COHORT FEATURES")
    print("="*80)
    
    cohort_features = cohort_df.rename(columns={'primaryid': 'PRIMARYID'})
    
    # Binary indicators for each cohort
    cohort_features['cohort_hbv_mono'] = (
        cohort_features['cohort'] == 'HBV_MONO'
    ).astype(int)
    cohort_features['cohort_hiv_hbv'] = (
        cohort_features['cohort'] == 'HIV_HBV_COINFECTION'
    ).astype(int)
    cohort_features['cohort_autoimmune'] = (
        cohort_features['cohort'] == 'HBV_AUTOIMMUNE'
    ).astype(int)
    
    print(f"  Extracted cohort for {len(cohort_features):,} patients")
    print(f"\n  Cohort distribution:")
    print(cohort_features['cohort'].value_counts())
    
    return cohort_features


def extract_drug_features(pairs_df, signals_df):
    """
    Extract drug-specific features including risk scores from Phase 2
    """
    print("\n" + "="*80)
    print("EXTRACTING DRUG FEATURES")
    print("="*80)
    
    # Get unique patient-drug combinations
    patient_drugs = pairs_df[['primaryid', 'drug_class']].drop_duplicates()
    patient_drugs = patient_drugs.rename(columns={'primaryid': 'PRIMARYID'})
    
    # Binary indicators for each drug
    for drug in ['ENTECAVIR', 'TENOFOVIR', 'LAMIVUDINE', 'ADEFOVIR', 'TELBIVUDINE', 'INTERFERON']:
        patient_drugs[f'drug_{drug.lower()}'] = (
            patient_drugs['drug_class'] == drug
        ).astype(int)
    
    # Drug risk scores from Phase 2 signals
    # Calculate mean ROR for each drug across all ADRs
    drug_risk_scores = signals_df.groupby('drug')['ror'].mean().to_dict()
    
    patient_drugs['drug_risk_score'] = patient_drugs['drug_class'].map(drug_risk_scores)
    patient_drugs['drug_risk_score'] = patient_drugs['drug_risk_score'].fillna(1.0)
    
    print(f"  Extracted drug features for {len(patient_drugs):,} patient-drug pairs")
    print(f"\n  Drug distribution:")
    print(patient_drugs['drug_class'].value_counts())
    print(f"\n  Drug risk scores:")
    for drug, score in drug_risk_scores.items():
        print(f"    {drug}: {score:.2f}")
    
    return patient_drugs


def extract_adr_labels(pairs_df):
    """
    Extract multi-label ADR categories for each patient
    """
    print("\n" + "="*80)
    print("EXTRACTING ADR LABELS")
    print("="*80)
    
    # Initialize label matrix
    patients = pairs_df['primaryid'].unique()
    labels_df = pd.DataFrame({'PRIMARYID': patients})
    
    # For each ADR category, check if patient has any ADR in that category
    for category, adr_list in config.ADR_CATEGORIES.items():
        # Find all ADRs in this category for all patients
        category_cases = pairs_df[pairs_df['adr'].str.upper().isin(
            [adr.upper() for adr in adr_list]
        )]
        
        # Mark patients who have this category
        patients_with_category = category_cases['primaryid'].unique()
        labels_df[f'adr_{category}'] = labels_df['PRIMARYID'].isin(
            patients_with_category
        ).astype(int)
    
    print(f"  Created labels for {len(labels_df):,} patients")
    print(f"\n  ADR category prevalence:")
    for category in config.ADR_CATEGORIES.keys():
        prevalence = labels_df[f'adr_{category}'].mean()
        count = labels_df[f'adr_{category}'].sum()
        print(f"    {category}: {prevalence:.1%} ({count:,} patients)")
    
    return labels_df

# ============================================================================
# FEATURE MATRIX CONSTRUCTION
# ============================================================================

def build_feature_matrix(data):
    """
    Combine all features into a single matrix for ML
    """
    print("\n" + "="*80)
    print("BUILDING COMPLETE FEATURE MATRIX")
    print("="*80)
    
    # Extract all feature sets
    demo_features = extract_demographics(data['demo'])
    severity_features = extract_severity_indicators(data['outc'])
    poly_features = extract_polypharmacy(data['drug'])
    cohort_features = extract_cohort_features(data['cohorts'])
    drug_features = extract_drug_features(data['pairs'], data['signals'])
    adr_labels = extract_adr_labels(data['pairs'])
    
    # CRITICAL FIX: Convert all PRIMARYID columns to string to avoid merge errors
    print("\n  Converting PRIMARYID to string type...")
    demo_features['PRIMARYID'] = demo_features['PRIMARYID'].astype(str)
    severity_features['PRIMARYID'] = severity_features['PRIMARYID'].astype(str)
    poly_features['PRIMARYID'] = poly_features['PRIMARYID'].astype(str)
    cohort_features['PRIMARYID'] = cohort_features['PRIMARYID'].astype(str)
    drug_features['PRIMARYID'] = drug_features['PRIMARYID'].astype(str)
    adr_labels['PRIMARYID'] = adr_labels['PRIMARYID'].astype(str)
    
    # Merge all features
    print("\n  Merging feature sets...")
    
    # Start with demographics
    feature_matrix = demo_features.copy()
    
    # Merge severity
    feature_matrix = feature_matrix.merge(
        severity_features, on='PRIMARYID', how='left'
    )
    
    # Merge polypharmacy
    feature_matrix = feature_matrix.merge(
        poly_features, on='PRIMARYID', how='left'
    )
    
    # Merge cohort
    feature_matrix = feature_matrix.merge(
        cohort_features[['PRIMARYID', 'cohort', 'cohort_hbv_mono', 
                          'cohort_hiv_hbv', 'cohort_autoimmune']], 
        on='PRIMARYID', how='left'
    )
    
    # Merge drug features
    feature_matrix = feature_matrix.merge(
        drug_features, on='PRIMARYID', how='left'
    )
    
    # Merge ADR labels
    feature_matrix = feature_matrix.merge(
        adr_labels, on='PRIMARYID', how='left'
    )
    
    # Fill missing values
    feature_matrix = feature_matrix.fillna(0)
    
    print(f"\n  ✓ Feature matrix shape: {feature_matrix.shape}")
    print(f"    Patients: {len(feature_matrix):,}")
    print(f"    Features: {feature_matrix.shape[1] - 6}")  # Exclude ID and label columns
    
    # Save feature matrix
    output_path = os.path.join(config.OUTPUT_DIR, 'feature_matrix.csv')
    feature_matrix.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved feature matrix: {output_path}")
    
    # Generate feature summary
    generate_feature_summary(feature_matrix)
    
    return feature_matrix


def generate_feature_summary(feature_matrix):
    """Generate summary of extracted features"""
    summary = []
    summary.append("\n" + "="*80)
    summary.append("FEATURE EXTRACTION SUMMARY")
    summary.append("="*80 + "\n")
    
    summary.append(f"Total Patients: {len(feature_matrix):,}\n")
    
    # Feature groups
    summary.append("FEATURE GROUPS:")
    summary.append("\n1. Demographics:")
    summary.append(f"   - Age (continuous + 4 categories)")
    summary.append(f"   - Sex (Male/Female/Unknown)")
    summary.append(f"   - Region (North America/Europe/Asia/Other)")
    summary.append(f"   - Reporter type (MD/Lawyer/Consumer)")
    
    summary.append("\n2. Disease Severity:")
    summary.append(f"   - Death indicator")
    summary.append(f"   - Life-threatening indicator")
    summary.append(f"   - Hospitalization indicator")
    summary.append(f"   - Severe disease score (0-5)")
    
    summary.append("\n3. Polypharmacy:")
    summary.append(f"   - Drug count (continuous)")
    summary.append(f"   - Polypharmacy categories (1-2 / 3-5 / 6+ drugs)")
    
    summary.append("\n4. Treatment Context:")
    summary.append(f"   - Cohort (HBV_MONO / HIV_HBV / AUTOIMMUNE)")
    summary.append(f"   - Cohort binary indicators")
    
    summary.append("\n5. Drug Features:")
    summary.append(f"   - Drug class (5 drugs)")
    summary.append(f"   - Drug risk score (from Phase 2 ROR)")
    
    summary.append("\n6. Target Labels (ADR Categories):")
    for category in config.ADR_CATEGORIES.keys():
        count = feature_matrix[f'adr_{category}'].sum()
        prevalence = feature_matrix[f'adr_{category}'].mean()
        summary.append(f"   - {category}: {count:,} ({prevalence:.1%})")
    
    summary.append("\n" + "="*80)
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    # Save summary
    summary_path = os.path.join(config.OUTPUT_DIR, 'feature_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"\n✓ Summary saved: {summary_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_feature_extraction():
    """Main execution function"""
    
    print("\n" + "🚀"*40)
    print("PHASE 3A: FEATURE EXTRACTION")
    print("🚀"*40)
    
    # Load data
    data = load_phase2_data()
    
    if data is None:
        print("\n❌ Error: Could not load required data files")
        return None
    
    # Build feature matrix
    feature_matrix = build_feature_matrix(data)
    
    print("\n" + "✅"*40)
    print("FEATURE EXTRACTION COMPLETE!")
    print("✅"*40)
    print(f"\nOutput saved to: {config.OUTPUT_DIR}")
    
    return feature_matrix


if __name__ == "__main__":
    feature_matrix = run_feature_extraction()