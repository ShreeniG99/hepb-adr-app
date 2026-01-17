# phase2_antiviral_adr_analysis.py

"""
Phase 2: Antiviral-ADR Pair Extraction and Signal Detection
- Stratify cohorts: HBV mono-infection vs HIV/HBV co-infection
- Extract drug-ADR pairs for Hepatitis B antivirals
- Perform disproportionality analysis (ROR, PRR, IC)
- Severity classification
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Phase2Config:
    """Configuration for Phase 2 analysis"""
    
    # Paths - UPDATE THESE TO MATCH YOUR SETUP
    DATA_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\hepatitis_b_data"
    OUTPUT_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase2_analysis"
    
    # Hepatitis B-specific antivirals
    HBV_ANTIVIRALS = {
        'ENTECAVIR': ['ENTECAVIR', 'BARACLUDE'],
        'TENOFOVIR': ['TENOFOVIR', 'VIREAD', 'VEMLIDY', 'TENOFOVIR DISOPROXIL', 
                      'TENOFOVIR ALAFENAMIDE', 'TDF', 'TAF'],
        'LAMIVUDINE': ['LAMIVUDINE', 'EPIVIR', '3TC'],
        'ADEFOVIR': ['ADEFOVIR', 'HEPSERA'],
        'TELBIVUDINE': ['TELBIVUDINE', 'TYZEKA'],
        'INTERFERON': ['PEGINTERFERON ALFA', 'INTERFERON ALFA', 'PEGASYS']
    }
    
    # HIV antiretrovirals (for co-infection detection)
    HIV_DRUGS = [
        'TRUVADA', 'ATRIPLA', 'BIKTARVY', 'STRIBILD', 'DESCOVY', 'GENVOYA',
        'COMPLERA', 'ODEFSEY', 'TRIUMEQ', 'DOVATO', 'TIVICAY', 'ISENTRESS',
        'PREZISTA', 'NORVIR', 'KALETRA', 'EVOTAZ', 'PREZCOBIX',
        'EFAVIRENZ', 'NEVIRAPINE', 'RILPIVIRINE', 'ETRAVIRINE',
        'RALTEGRAVIR', 'DOLUTEGRAVIR', 'ELVITEGRAVIR', 'BICTEGRAVIR',
        'DARUNAVIR', 'ATAZANAVIR', 'LOPINAVIR', 'RITONAVIR',
        'EMTRICITABINE', 'ABACAVIR', 'ZIDOVUDINE', 'DIDANOSINE'
    ]
    
    # Immunosuppressants (for autoimmune co-morbidity detection)
    IMMUNOSUPPRESSANTS = [
        'METHOTREXATE', 'SULFASALAZINE', 'RITUXIMAB', 'ACTEMRA', 'ORENCIA',
        'HUMIRA', 'ENBREL', 'REMICADE', 'PREDNISONE', 'AZATHIOPRINE',
        'CYCLOSPORINE', 'TACROLIMUS', 'MYCOPHENOLATE'
    ]
    
    # Signal detection thresholds
    SIGNAL_THRESHOLDS = {
        'ROR_LOWER_CI': 2.0,      # Lower bound of 95% CI for ROR
        'PRR': 2.0,                # Proportional Reporting Ratio
        'IC_LOWER_CI': 0.0,        # Lower bound of IC025
        'MIN_CASES': 3             # Minimum number of cases
    }
    
    # Severity classification (based on outcomes)
    SEVERITY_MAPPING = {
        'DE': 5,  # Death
        'LT': 4,  # Life-threatening
        'HO': 3,  # Hospitalization
        'DS': 3,  # Disability
        'CA': 2,  # Congenital Anomaly
        'RI': 2,  # Required Intervention
        'OT': 1   # Other
    }

config = Phase2Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_extracted_data():
    """Load the extracted Hepatitis B data from Phase 1"""
    print("\n" + "="*80)
    print("LOADING PHASE 1 DATA")
    print("="*80)
    
    data = {}
    
    files = {
        'demo': 'hepb_demo.csv',
        'drug': 'hepb_drug.csv',
        'reac': 'hepb_reac.csv',
        'indi': 'hepb_indi.csv',
        'outc': 'hepb_outc.csv'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(config.DATA_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, low_memory=False, dtype=str)
            data[key] = df
            print(f"✓ Loaded {key.upper()}: {len(df):,} records")
        else:
            print(f"✗ Missing {key.upper()}: {filepath}")
            data[key] = None
    
    return data

# ============================================================================
# COHORT STRATIFICATION
# ============================================================================

def identify_drug_exposure(drug_df):
    """
    Identify patients exposed to different drug classes
    Returns dict of {primaryid: set of drug classes}
    """
    print("\n" + "="*80)
    print("IDENTIFYING DRUG EXPOSURES")
    print("="*80)
    
    drug_df['DRUGNAME_UPPER'] = drug_df['DRUGNAME'].str.upper()
    if 'PROD_AI' in drug_df.columns:
        drug_df['PROD_AI_UPPER'] = drug_df['PROD_AI'].str.upper()
    else:
        drug_df['PROD_AI_UPPER'] = ''
    
    patient_drugs = {}
    
    # Identify HBV antiviral exposure
    hbv_patients = set()
    hbv_by_drug = {}
    
    for drug_class, drug_names in config.HBV_ANTIVIRALS.items():
        patients = set()
        for drug in drug_names:
            mask = (
                drug_df['DRUGNAME_UPPER'].str.contains(drug, na=False, regex=False) |
                drug_df['PROD_AI_UPPER'].str.contains(drug, na=False, regex=False)
            )
            patients.update(drug_df[mask]['PRIMARYID'].unique())
        
        hbv_by_drug[drug_class] = patients
        hbv_patients.update(patients)
        print(f"  {drug_class}: {len(patients):,} patients")
    
    print(f"\n  Total HBV antiviral exposure: {len(hbv_patients):,} patients")
    
    # Identify HIV drug exposure
    hiv_patients = set()
    for drug in config.HIV_DRUGS:
        mask = (
            drug_df['DRUGNAME_UPPER'].str.contains(drug, na=False, regex=False) |
            drug_df['PROD_AI_UPPER'].str.contains(drug, na=False, regex=False)
        )
        hiv_patients.update(drug_df[mask]['PRIMARYID'].unique())
    
    print(f"  HIV drug exposure: {len(hiv_patients):,} patients")
    
    # Identify immunosuppressant exposure
    immunosupp_patients = set()
    for drug in config.IMMUNOSUPPRESSANTS:
        mask = (
            drug_df['DRUGNAME_UPPER'].str.contains(drug, na=False, regex=False) |
            drug_df['PROD_AI_UPPER'].str.contains(drug, na=False, regex=False)
        )
        immunosupp_patients.update(drug_df[mask]['PRIMARYID'].unique())
    
    print(f"  Immunosuppressant exposure: {len(immunosupp_patients):,} patients")
    
    return {
        'hbv_patients': hbv_patients,
        'hbv_by_drug': hbv_by_drug,
        'hiv_patients': hiv_patients,
        'immunosupp_patients': immunosupp_patients
    }


def stratify_cohorts(data, drug_exposures):
    """
    Stratify patients into cohorts:
    1. HBV mono-infection (HBV antivirals only, no HIV drugs)
    2. HIV/HBV co-infection (both HBV and HIV drugs)
    3. HBV + Autoimmune (HBV + immunosuppressants, no HIV)
    4. Other HBV (Hepatitis B diagnosis but no HBV antivirals)
    """
    print("\n" + "="*80)
    print("STRATIFYING COHORTS")
    print("="*80)
    
    demo_df = data['demo']
    all_patients = set(demo_df['PRIMARYID'].unique())
    
    hbv_av = drug_exposures['hbv_patients']
    hiv = drug_exposures['hiv_patients']
    immunosupp = drug_exposures['immunosupp_patients']
    
    # Define cohorts
    cohorts = {}
    
    # Cohort 1: HBV mono-infection
    cohorts['HBV_MONO'] = hbv_av - hiv
    print(f"  Cohort 1 - HBV Mono-infection: {len(cohorts['HBV_MONO']):,} patients")
    
    # Cohort 2: HIV/HBV co-infection
    cohorts['HIV_HBV_COINFECTION'] = hbv_av & hiv
    print(f"  Cohort 2 - HIV/HBV Co-infection: {len(cohorts['HIV_HBV_COINFECTION']):,} patients")
    
    # Cohort 3: HBV + Autoimmune (no HIV)
    cohorts['HBV_AUTOIMMUNE'] = (hbv_av & immunosupp) - hiv
    print(f"  Cohort 3 - HBV + Autoimmune: {len(cohorts['HBV_AUTOIMMUNE']):,} patients")
    
    # Cohort 4: Other HBV (diagnosed but not on HBV antivirals)
    cohorts['HBV_OTHER'] = all_patients - hbv_av
    print(f"  Cohort 4 - Other HBV (no antivirals): {len(cohorts['HBV_OTHER']):,} patients")
    
    # Verify no overlap
    total_assigned = sum(len(cohort) for cohort in cohorts.values())
    print(f"\n  Total patients: {len(all_patients):,}")
    print(f"  Total assigned: {total_assigned:,}")
    
    # Save cohort assignments
    cohort_assignments = []
    for cohort_name, patients in cohorts.items():
        for patient_id in patients:
            cohort_assignments.append({
                'primaryid': patient_id,
                'cohort': cohort_name
            })
    
    cohort_df = pd.DataFrame(cohort_assignments)
    output_path = os.path.join(config.OUTPUT_DIR, 'cohort_assignments.csv')
    cohort_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved cohort assignments: {output_path}")
    
    return cohorts, cohort_df

# ============================================================================
# DRUG-ADR PAIR EXTRACTION
# ============================================================================

def extract_drug_adr_pairs(data, drug_exposures, cohorts):
    """
    Extract drug-ADR pairs for each HBV antiviral
    Link with severity outcomes
    """
    print("\n" + "="*80)
    print("EXTRACTING DRUG-ADR PAIRS")
    print("="*80)
    
    drug_df = data['drug'].copy()
    reac_df = data['reac'].copy()
    outc_df = data['outc'].copy()
    
    drug_df['DRUGNAME_UPPER'] = drug_df['DRUGNAME'].str.upper()
    if 'PROD_AI' in drug_df.columns:
        drug_df['PROD_AI_UPPER'] = drug_df['PROD_AI'].str.upper()
    else:
        drug_df['PROD_AI_UPPER'] = ''
    
    # Get severity for each case
    severity_map = {}
    for _, row in outc_df.iterrows():
        primaryid = row['PRIMARYID']
        outcome = row['OUTC_COD']
        severity = config.SEVERITY_MAPPING.get(outcome, 1)
        
        # Keep the maximum severity for each case
        if primaryid not in severity_map or severity > severity_map[primaryid]:
            severity_map[primaryid] = severity
    
    all_pairs = []
    
    # For each HBV antiviral class
    for drug_class, drug_names in config.HBV_ANTIVIRALS.items():
        print(f"\n  Processing {drug_class}...")
        
        # Get all cases exposed to this drug
        drug_mask = pd.Series(False, index=drug_df.index)
        for drug_name in drug_names:
            drug_mask |= (
                drug_df['DRUGNAME_UPPER'].str.contains(drug_name, na=False, regex=False) |
                drug_df['PROD_AI_UPPER'].str.contains(drug_name, na=False, regex=False)
            )
        
        drug_cases = drug_df[drug_mask]['PRIMARYID'].unique()
        print(f"    Cases exposed: {len(drug_cases):,}")
        
        # Get ADRs for these cases
        case_reactions = reac_df[reac_df['PRIMARYID'].isin(drug_cases)]
        print(f"    ADR records: {len(case_reactions):,}")
        
        # Determine cohort for each case
        for _, row in case_reactions.iterrows():
            primaryid = row['PRIMARYID']
            adr = row['PT']
            
            # Determine cohort
            cohort = 'UNKNOWN'
            for cohort_name, cohort_patients in cohorts.items():
                if primaryid in cohort_patients:
                    cohort = cohort_name
                    break
            
            # Get severity
            severity = severity_map.get(primaryid, 1)
            
            all_pairs.append({
                'primaryid': primaryid,
                'drug_class': drug_class,
                'adr': adr,
                'cohort': cohort,
                'severity': severity
            })
    
    pairs_df = pd.DataFrame(all_pairs)
    
    print(f"\n  Total drug-ADR pairs: {len(pairs_df):,}")
    print(f"  Unique ADRs: {pairs_df['adr'].nunique():,}")
    
    # Save pairs
    output_path = os.path.join(config.OUTPUT_DIR, 'drug_adr_pairs.csv')
    pairs_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved drug-ADR pairs: {output_path}")
    
    return pairs_df

# ============================================================================
# SIGNAL DETECTION - DISPROPORTIONALITY ANALYSIS
# ============================================================================

def calculate_ror(a, b, c, d):
    """
    Calculate Reporting Odds Ratio (ROR) with 95% CI
    
    a = cases with drug AND adr
    b = cases with drug but NOT adr
    c = cases with adr but NOT drug
    d = cases with neither drug nor adr
    """
    if a == 0 or b == 0 or c == 0 or d == 0:
        return None, None, None
    
    ror = (a * d) / (b * c)
    
    # 95% CI for log(ROR)
    se_log_ror = np.sqrt(1/a + 1/b + 1/c + 1/d)
    log_ror = np.log(ror)
    
    ci_lower = np.exp(log_ror - 1.96 * se_log_ror)
    ci_upper = np.exp(log_ror + 1.96 * se_log_ror)
    
    return ror, ci_lower, ci_upper


def calculate_prr(a, b, c, d):
    """
    Calculate Proportional Reporting Ratio (PRR) with 95% CI
    """
    if (a + c) == 0 or (b + d) == 0:
        return None, None, None
    
    prr = (a / (a + b)) / (c / (c + d))
    
    # Standard error
    se_log_prr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
    log_prr = np.log(prr)
    
    ci_lower = np.exp(log_prr - 1.96 * se_log_prr)
    ci_upper = np.exp(log_prr + 1.96 * se_log_prr)
    
    return prr, ci_lower, ci_upper


def calculate_ic(a, b, c, d):
    """
    Calculate Information Component (IC) with 95% CI
    Bayesian approach for signal detection
    """
    n = a + b + c + d
    
    if a == 0 or n == 0:
        return None, None, None
    
    # Expected count under independence
    expected = ((a + b) * (a + c)) / n
    
    if expected == 0:
        return None, None, None
    
    # IC = log2(observed / expected)
    ic = np.log2(a / expected)
    
    # Approximate 95% CI (simplified Bayesian credible interval)
    ic_025 = ic - 1.96 * np.sqrt(1/a)
    ic_975 = ic + 1.96 * np.sqrt(1/a)
    
    return ic, ic_025, ic_975


def perform_signal_detection(pairs_df, cohort_name=None):
    """
    Perform disproportionality analysis for drug-ADR pairs
    
    Args:
        pairs_df: DataFrame with drug-ADR pairs
        cohort_name: If specified, filter to this cohort only
    """
    if cohort_name:
        print(f"\n{'='*80}")
        print(f"SIGNAL DETECTION: {cohort_name}")
        print(f"{'='*80}")
        pairs = pairs_df[pairs_df['cohort'] == cohort_name].copy()
    else:
        print(f"\n{'='*80}")
        print(f"SIGNAL DETECTION: ALL COHORTS")
        print(f"{'='*80}")
        pairs = pairs_df.copy()
    
    if len(pairs) == 0:
        print(f"  No data for {cohort_name}")
        return None
    
    print(f"  Total pairs: {len(pairs):,}")
    
    # Get unique drugs and ADRs
    drugs = pairs['drug_class'].unique()
    adrs = pairs['adr'].unique()
    
    print(f"  Unique drugs: {len(drugs)}")
    print(f"  Unique ADRs: {len(adrs):,}")
    
    signals = []
    
    total_cases = pairs['primaryid'].nunique()
    
    for drug in drugs:
        drug_pairs = pairs[pairs['drug_class'] == drug]
        drug_cases = drug_pairs['primaryid'].nunique()
        
        for adr in adrs:
            # Construct 2x2 contingency table
            # a = cases with BOTH drug AND adr
            a = len(drug_pairs[drug_pairs['adr'] == adr]['primaryid'].unique())
            
            if a < config.SIGNAL_THRESHOLDS['MIN_CASES']:
                continue
            
            # b = cases with drug but NOT adr
            b = drug_cases - a
            
            # c = cases with adr but NOT drug
            adr_pairs = pairs[pairs['adr'] == adr]
            adr_cases = adr_pairs['primaryid'].nunique()
            c = adr_cases - a
            
            # d = cases with neither drug nor adr
            d = total_cases - (a + b + c)
            
            if b <= 0 or c <= 0 or d <= 0:
                continue
            
            # Calculate metrics
            ror, ror_lower, ror_upper = calculate_ror(a, b, c, d)
            prr, prr_lower, prr_upper = calculate_prr(a, b, c, d)
            ic, ic_025, ic_975 = calculate_ic(a, b, c, d)
            
            if ror is None or prr is None or ic is None:
                continue
            
            # Check signal criteria
            is_signal = (
                ror_lower >= config.SIGNAL_THRESHOLDS['ROR_LOWER_CI'] and
                prr >= config.SIGNAL_THRESHOLDS['PRR'] and
                ic_025 >= config.SIGNAL_THRESHOLDS['IC_LOWER_CI']
            )
            
            # Calculate average severity for this drug-ADR pair
            severity_cases = drug_pairs[drug_pairs['adr'] == adr]
            avg_severity = severity_cases['severity'].astype(float).mean()
            
            signals.append({
                'drug': drug,
                'adr': adr,
                'cohort': cohort_name if cohort_name else 'ALL',
                'n_cases': a,
                'ror': ror,
                'ror_lower_ci': ror_lower,
                'ror_upper_ci': ror_upper,
                'prr': prr,
                'prr_lower_ci': prr_lower,
                'prr_upper_ci': prr_upper,
                'ic': ic,
                'ic_025': ic_025,
                'ic_975': ic_975,
                'is_signal': is_signal,
                'avg_severity': avg_severity
            })
    
    signals_df = pd.DataFrame(signals)
    
    if len(signals_df) > 0:
        # Sort by ROR descending
        signals_df = signals_df.sort_values('ror', ascending=False)
        
        n_signals = signals_df['is_signal'].sum()
        print(f"\n  Total drug-ADR combinations analyzed: {len(signals_df):,}")
        print(f"  Detected signals: {n_signals:,}")
        
        # Show top 10 signals
        top_signals = signals_df[signals_df['is_signal']].head(10)
        if len(top_signals) > 0:
            print(f"\n  Top 10 Signals:")
            for _, row in top_signals.iterrows():
                print(f"    {row['drug']} + {row['adr']}: ROR={row['ror']:.2f} (n={row['n_cases']})")
    
    return signals_df

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def compare_cohorts(pairs_df):
    """
    Compare ADR profiles between HBV mono-infection and HIV/HBV co-infection
    """
    print("\n" + "="*80)
    print("COMPARATIVE COHORT ANALYSIS")
    print("="*80)
    
    # Focus on the two main cohorts
    cohort1 = pairs_df[pairs_df['cohort'] == 'HBV_MONO']
    cohort2 = pairs_df[pairs_df['cohort'] == 'HIV_HBV_COINFECTION']
    
    print(f"\n  HBV Mono-infection: {len(cohort1):,} ADR records")
    print(f"  HIV/HBV Co-infection: {len(cohort2):,} ADR records")
    
    # Compare ADR frequencies
    comparison = []
    
    all_adrs = set(cohort1['adr'].unique()) | set(cohort2['adr'].unique())
    
    for adr in all_adrs:
        c1_count = len(cohort1[cohort1['adr'] == adr])
        c2_count = len(cohort2[cohort2['adr'] == adr])
        
        c1_total = len(cohort1)
        c2_total = len(cohort2)
        
        if c1_total == 0 or c2_total == 0:
            continue
        
        c1_prop = c1_count / c1_total
        c2_prop = c2_count / c2_total
        
        # Risk ratio
        if c2_prop > 0:
            risk_ratio = c1_prop / c2_prop
        else:
            risk_ratio = np.nan
        
        # Chi-square test for independence
        if c1_count >= 5 and c2_count >= 5:
            contingency = np.array([
                [c1_count, c1_total - c1_count],
                [c2_count, c2_total - c2_count]
            ])
            chi2, p_value = stats.chi2_contingency(contingency)[:2]
        else:
            chi2 = np.nan
            p_value = np.nan
        
        comparison.append({
            'adr': adr,
            'hbv_mono_count': c1_count,
            'hbv_mono_prop': c1_prop,
            'hiv_hbv_count': c2_count,
            'hiv_hbv_prop': c2_prop,
            'risk_ratio': risk_ratio,
            'chi2': chi2,
            'p_value': p_value
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('risk_ratio', ascending=False)
    
    # Save comparison
    output_path = os.path.join(config.OUTPUT_DIR, 'cohort_comparison.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved cohort comparison: {output_path}")
    
    # Show top differences
    print("\n  Top 10 ADRs MORE common in HBV Mono-infection:")
    top_hbv = comparison_df.nlargest(10, 'risk_ratio')
    for _, row in top_hbv.iterrows():
        if not np.isnan(row['risk_ratio']):
            print(f"    {row['adr']}: RR={row['risk_ratio']:.2f} (p={row['p_value']:.4f})")
    
    print("\n  Top 10 ADRs MORE common in HIV/HBV Co-infection:")
    top_hiv = comparison_df.nsmallest(10, 'risk_ratio')
    for _, row in top_hiv.iterrows():
        if not np.isnan(row['risk_ratio']):
            print(f"    {row['adr']}: RR={row['risk_ratio']:.2f} (p={row['p_value']:.4f})")
    
    return comparison_df

# ============================================================================
# SEVERITY ANALYSIS
# ============================================================================

def analyze_severity_by_cohort(pairs_df):
    """
    Analyze ADR severity across cohorts
    """
    print("\n" + "="*80)
    print("SEVERITY ANALYSIS BY COHORT")
    print("="*80)
    
    severity_summary = []
    
    for cohort in pairs_df['cohort'].unique():
        cohort_pairs = pairs_df[pairs_df['cohort'] == cohort]
        
        if len(cohort_pairs) == 0:
            continue
        
        # Get unique cases per cohort
        n_cases = cohort_pairs['primaryid'].nunique()
        
        # Count severity levels
        severity_counts = cohort_pairs.groupby('severity').size()
        
        severity_summary.append({
            'cohort': cohort,
            'n_cases': n_cases,
            'n_adr_records': len(cohort_pairs),
            'mean_severity': cohort_pairs['severity'].astype(float).mean(),
            'median_severity': cohort_pairs['severity'].astype(float).median(),
            'severity_1_count': severity_counts.get(1, 0),
            'severity_2_count': severity_counts.get(2, 0),
            'severity_3_count': severity_counts.get(3, 0),
            'severity_4_count': severity_counts.get(4, 0),
            'severity_5_count': severity_counts.get(5, 0)
        })
    
    severity_df = pd.DataFrame(severity_summary)
    
    # Display
    print("\n  Severity by Cohort:")
    for _, row in severity_df.iterrows():
        print(f"\n  {row['cohort']}:")
        print(f"    Cases: {row['n_cases']:,}")
        print(f"    Mean severity: {row['mean_severity']:.2f}")
        print(f"    Deaths (severity 5): {row['severity_5_count']:,}")
        print(f"    Life-threatening (severity 4): {row['severity_4_count']:,}")
        print(f"    Hospitalization (severity 3): {row['severity_3_count']:,}")
    
    # Save
    output_path = os.path.join(config.OUTPUT_DIR, 'severity_by_cohort.csv')
    severity_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved severity analysis: {output_path}")
    
    return severity_df

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_phase2_pipeline():
    """
    Main Phase 2 pipeline
    """
   
    print("PHASE 2: ANTIVIRAL-ADR ANALYSIS")
    
    
    # Step 1: Load data
    data = load_extracted_data()
    
    # Step 2: Identify drug exposures
    drug_exposures = identify_drug_exposure(data['drug'])
    
    # Step 3: Stratify cohorts
    cohorts, cohort_df = stratify_cohorts(data, drug_exposures)
    
    # Step 4: Extract drug-ADR pairs
    pairs_df = extract_drug_adr_pairs(data, drug_exposures, cohorts)
    
    # Step 5: Signal detection for each cohort
    all_signals = []
    
    for cohort_name in ['HBV_MONO', 'HIV_HBV_COINFECTION', 'HBV_AUTOIMMUNE']:
        signals = perform_signal_detection(pairs_df, cohort_name)
        if signals is not None:
            all_signals.append(signals)
    
    # Combine all signals
    if all_signals:
        combined_signals = pd.concat(all_signals, ignore_index=True)
        output_path = os.path.join(config.OUTPUT_DIR, 'all_signals.csv')
        combined_signals.to_csv(output_path, index=False)
        print(f"\n✓ Saved combined signals: {output_path}")
    
    # Step 6: Comparative cohort analysis
    comparison_df = compare_cohorts(pairs_df)
    
    # Step 7: Severity analysis
    severity_df = analyze_severity_by_cohort(pairs_df)
    
    # Step 8: Generate summary report
    generate_phase2_summary(cohorts, pairs_df, combined_signals if all_signals else None)
    
    print("\n" + "✅"*40)
    print("PHASE 2 COMPLETE!")
    print("✅"*40)
    
    return {
        'cohorts': cohorts,
        'pairs': pairs_df,
        'signals': combined_signals if all_signals else None,
        'comparison': comparison_df,
        'severity': severity_df
    }


def generate_phase2_summary(cohorts, pairs_df, signals_df):
    """
    Generate Phase 2 summary report
    """
    report = []
    report.append("\n" + "="*80)
    report.append("PHASE 2 ANALYSIS SUMMARY")
    report.append("="*80 + "\n")
    
    # Cohort sizes
    report.append("COHORT STRATIFICATION:")
    for cohort_name, patients in cohorts.items():
        report.append(f"  {cohort_name}: {len(patients):,} patients")
    report.append("")
    
    # Drug-ADR pairs by cohort
    report.append("DRUG-ADR PAIRS BY COHORT:")
    for cohort in pairs_df['cohort'].unique():
        cohort_pairs = pairs_df[pairs_df['cohort'] == cohort]
        n_cases = cohort_pairs['primaryid'].nunique()
        n_adrs = cohort_pairs['adr'].nunique()
        report.append(f"  {cohort}:")
        report.append(f"    Cases: {n_cases:,}")
        report.append(f"    ADR records: {len(cohort_pairs):,}")
        report.append(f"    Unique ADRs: {n_adrs:,}")
    report.append("")
    
    # Signal detection summary
    if signals_df is not None:
        report.append("SIGNAL DETECTION SUMMARY:")
        for cohort in signals_df['cohort'].unique():
            cohort_signals = signals_df[signals_df['cohort'] == cohort]
            n_signals = cohort_signals['is_signal'].sum()
            report.append(f"  {cohort}: {n_signals:,} significant signals detected")
        
        report.append("\n  Top 10 Strongest Signals (by ROR):")
        top_signals = signals_df[signals_df['is_signal']].nlargest(10, 'ror')
        for _, row in top_signals.iterrows():
            report.append(f"    {row['drug']} + {row['adr']} ({row['cohort']})")
            report.append(f"      ROR: {row['ror']:.2f} (95% CI: {row['ror_lower_ci']:.2f}-{row['ror_upper_ci']:.2f})")
            report.append(f"      Cases: {row['n_cases']}, Severity: {row['avg_severity']:.2f}")
    
    report.append("\n" + "="*80)
    
    summary_text = "\n".join(report)
    print(summary_text)
    
    # Save
    summary_path = os.path.join(config.OUTPUT_DIR, 'phase2_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\n✓ Summary saved: {summary_path}")


# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    results = run_phase2_pipeline()