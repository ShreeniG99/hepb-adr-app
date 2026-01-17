# hepatitis_b_extraction_fixed.py

import os
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class FAERSConfig:
    """Configuration for FAERS data extraction"""
    
    BASE_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\Datasets"
    RAW_DATA_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\Datasets\extracted"
    OUTPUT_DIR = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\hepatitis_b_data"
    
    START_YEAR = 2020
    END_YEAR = 2025
    
    DELIMITER = '$'
    ENCODING = 'ISO-8859-1'
    
    # MedDRA Preferred Terms for Hepatitis B
    HEPATITIS_B_TERMS = [
        'HEPATITIS B',
        'CHRONIC HEPATITIS B',
        'HEPATITIS B ACUTE',
        'HEPATITIS B CARRIER',
        'HEPATITIS B REACTIVATION',
        'HEPATITIS B ANTIGEN POSITIVE',
        'HEPATITIS B SURFACE ANTIGEN POSITIVE',
        'HEPATITIS B E ANTIGEN POSITIVE',
        'HEPATITIS B ANTIBODY POSITIVE',
        'HEPATITIS B VIRUS INFECTION',
        'HEPATOCELLULAR CARCINOMA',
        'HEPATIC CIRRHOSIS',
        'LIVER DISORDER',
        'HEPATIC FAILURE',
        'HEPATIC ENCEPHALOPATHY'
    ]
    
    # Antiviral drugs for Hepatitis B
    HEPATITIS_B_DRUGS = [
        'ENTECAVIR',
        'TENOFOVIR',
        'LAMIVUDINE',
        'ADEFOVIR',
        'TELBIVUDINE',
        'EMTRICITABINE',
        'BARACLUDE',
        'VIREAD',
        'EPIVIR',
        'HEPSERA',
        'TYZEKA',
        'EMTRIVA',
        'VEMLIDY',
        'TENOFOVIR DISOPROXIL',
        'TENOFOVIR ALAFENAMIDE',
        'PEGINTERFERON ALFA',
        'INTERFERON ALFA'
    ]
    
    DEMO_COLS = ['primaryid', 'caseid', 'caseversion', 'i_f_code', 'event_dt', 
                  'age', 'age_cod', 'sex', 'wt', 'wt_cod', 'occp_cod', 'reporter_country']
    
    DRUG_COLS = ['primaryid', 'caseid', 'drug_seq', 'role_cod', 'drugname', 
                  'prod_ai', 'val_vbm', 'route', 'dose_vbm', 'dechal', 'rechal']
    
    REAC_COLS = ['primaryid', 'caseid', 'pt', 'drug_rec_act']
    
    INDI_COLS = ['primaryid', 'caseid', 'indi_drug_seq', 'indi_pt']
    
    OUTC_COLS = ['primaryid', 'caseid', 'outc_cod']
    
    RPSR_COLS = ['primaryid', 'caseid', 'rpsr_cod']

config = FAERSConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FILE DISCOVERY AND LOADING
# ============================================================================

def find_faers_files(base_dir, year, quarter, file_type):
    """Find FAERS files for a specific quarter"""
    year_short = str(year)[2:]
    
    patterns = [
        f"*{file_type}{year_short}q{quarter}*.txt",
        f"*{file_type}{year_short}Q{quarter}*.txt",
        f"*{file_type.lower()}{year_short}q{quarter}*.txt",
        f"*{file_type.lower()}{year_short}Q{quarter}*.txt",
    ]
    
    for pattern in patterns:
        matches = list(Path(base_dir).rglob(pattern))
        if matches:
            return str(matches[0])
    
    return None


def load_faers_file(filepath, columns_to_keep=None):
    """Load a FAERS data file"""
    if filepath is None or not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(
            filepath,
            sep=config.DELIMITER,
            encoding=config.ENCODING,
            low_memory=False,
            dtype=str,
            on_bad_lines='skip'
        )
        
        df.columns = df.columns.str.upper().str.strip()
        
        if columns_to_keep:
            available_cols = [col for col in columns_to_keep if col.upper() in df.columns]
            if available_cols:
                df = df[[col.upper() for col in available_cols]]
        
        df = df.dropna(how='all')
        
        return df
    
    except Exception as e:
        print(f"  ✗ Error loading {os.path.basename(filepath)}: {e}")
        return None


# ============================================================================
# QUARTER-BY-QUARTER IDENTIFICATION (MEMORY EFFICIENT)
# ============================================================================

def identify_hepb_cases_in_quarter(year, quarter, base_dir):
    """
    Identify Hepatitis B cases in a single quarter
    Returns set of primaryids
    """
    hepb_cases = set()
    
    # Method 1: Check INDI file
    indi_path = find_faers_files(base_dir, year, quarter, 'INDI')
    if indi_path:
        try:
            indi_df = load_faers_file(indi_path, config.INDI_COLS)
            if indi_df is not None:
                indi_df['INDI_PT_UPPER'] = indi_df['INDI_PT'].str.upper()
                for term in config.HEPATITIS_B_TERMS:
                    mask = indi_df['INDI_PT_UPPER'].str.contains(term, na=False, regex=False)
                    hepb_cases.update(indi_df[mask]['PRIMARYID'].unique())
                del indi_df
                gc.collect()
        except Exception as e:
            print(f"    Warning: INDI error - {e}")
    
    # Method 2: Check REAC file
    reac_path = find_faers_files(base_dir, year, quarter, 'REAC')
    if reac_path:
        try:
            reac_df = load_faers_file(reac_path, config.REAC_COLS)
            if reac_df is not None:
                reac_df['PT_UPPER'] = reac_df['PT'].str.upper()
                for term in config.HEPATITIS_B_TERMS:
                    mask = reac_df['PT_UPPER'].str.contains(term, na=False, regex=False)
                    hepb_cases.update(reac_df[mask]['PRIMARYID'].unique())
                del reac_df
                gc.collect()
        except Exception as e:
            print(f"    Warning: REAC error - {e}")
    
    # Method 3: Check DRUG file
    drug_path = find_faers_files(base_dir, year, quarter, 'DRUG')
    if drug_path:
        try:
            drug_df = load_faers_file(drug_path, config.DRUG_COLS)
            if drug_df is not None:
                drug_df['DRUGNAME_UPPER'] = drug_df['DRUGNAME'].str.upper()
                if 'PROD_AI' in drug_df.columns:
                    drug_df['PROD_AI_UPPER'] = drug_df['PROD_AI'].str.upper()
                else:
                    drug_df['PROD_AI_UPPER'] = ''
                
                for drug in config.HEPATITIS_B_DRUGS:
                    mask = (
                        drug_df['DRUGNAME_UPPER'].str.contains(drug, na=False, regex=False) |
                        drug_df['PROD_AI_UPPER'].str.contains(drug, na=False, regex=False)
                    )
                    hepb_cases.update(drug_df[mask]['PRIMARYID'].unique())
                del drug_df
                gc.collect()
        except Exception as e:
            print(f"    Warning: DRUG error - {e}")
    
    return hepb_cases


def extract_quarter_data(year, quarter, base_dir, hepb_cases):
    """
    Extract data for Hepatitis B cases from a single quarter
    """
    quarter_data = {}
    
    file_types = {
        'demo': config.DEMO_COLS,
        'drug': config.DRUG_COLS,
        'reac': config.REAC_COLS,
        'indi': config.INDI_COLS,
        'outc': config.OUTC_COLS,
        'rpsr': config.RPSR_COLS
    }
    
    for file_type, cols in file_types.items():
        file_path = find_faers_files(base_dir, year, quarter, file_type.upper())
        
        if file_path:
            df = load_faers_file(file_path, cols)
            if df is not None and len(df) > 0:
                # Filter immediately
                filtered_df = df[df['PRIMARYID'].isin(hepb_cases)].copy()
                if len(filtered_df) > 0:
                    filtered_df['year_quarter'] = f"{year} Q{quarter}"
                    quarter_data[file_type] = filtered_df
                del df
                gc.collect()
    
    return quarter_data


# ============================================================================
# MAIN EXTRACTION PIPELINE (MEMORY EFFICIENT)
# ============================================================================

def run_extraction_pipeline():
    """
    Memory-efficient extraction pipeline
    Process quarter-by-quarter
    """
    
    print(f"\n{'='*80}")
    print("PHASE 1: IDENTIFYING HEPATITIS B CASES QUARTER-BY-QUARTER")
    print(f"{'='*80}")
    
    all_hepb_cases = set()
    
    # Phase 1: Identify all Hepatitis B cases
    for year in range(config.START_YEAR, config.END_YEAR + 1):
        for quarter in range(1, 5):
            if year == 2025 and quarter > 4:
                break
            
            quarter_label = f"{year} Q{quarter}"
            print(f"\n{quarter_label}:")
            
            quarter_cases = identify_hepb_cases_in_quarter(year, quarter, config.RAW_DATA_DIR)
            
            if quarter_cases:
                all_hepb_cases.update(quarter_cases)
                print(f"  ✓ Found {len(quarter_cases):,} Hep B cases in this quarter")
                print(f"  ✓ Total cumulative: {len(all_hepb_cases):,} unique cases")
            else:
                print(f"  ✗ No Hep B cases found")
    
    print(f"\n{'='*80}")
    print(f"TOTAL HEPATITIS B CASES IDENTIFIED: {len(all_hepb_cases):,}")
    print(f"{'='*80}")
    
    if len(all_hepb_cases) == 0:
        print("\n No Hepatitis B cases found!")
        return None
    
    # Phase 2: Extract data quarter-by-quarter
    print(f"\n{'='*80}")
    print("PHASE 2: EXTRACTING DATA FOR HEPATITIS B CASES")
    print(f"{'='*80}")
    
    extracted_data = {
        'demo': [],
        'drug': [],
        'reac': [],
        'indi': [],
        'outc': [],
        'rpsr': []
    }
    
    for year in range(config.START_YEAR, config.END_YEAR + 1):
        for quarter in range(1, 5):
            if year == 2025 and quarter > 4:
                break
            
            quarter_label = f"{year} Q{quarter}"
            print(f"\n{quarter_label}:")
            
            quarter_data = extract_quarter_data(year, quarter, config.RAW_DATA_DIR, all_hepb_cases)
            
            for key, df in quarter_data.items():
                extracted_data[key].append(df)
                print(f"  ✓ {key.upper()}: {len(df):,} records")
            
            gc.collect()
    
    # Phase 3: Concatenate filtered data
    print(f"\n{'='*80}")
    print("PHASE 3: CONCATENATING FILTERED DATA")
    print(f"{'='*80}")
    
    hepb_data = {}
    for key, dfs in extracted_data.items():
        if dfs:
            hepb_data[key] = pd.concat(dfs, ignore_index=True)
            print(f"✓ {key.upper()}: {len(hepb_data[key]):,} total records")
        else:
            hepb_data[key] = None
            print(f"✗ {key.upper()}: No data")
    
    # Phase 4: Deduplicate
    print(f"\n{'='*80}")
    print("PHASE 4: DEDUPLICATING CASES")
    print(f"{'='*80}")
    
    hepb_data_dedup = deduplicate_cases(hepb_data)
    
    # Phase 5: Clean demographics
    print(f"\n{'='*80}")
    print("PHASE 5: CLEANING DEMOGRAPHIC DATA")
    print(f"{'='*80}")
    
    if hepb_data_dedup['demo'] is not None:
        hepb_data_dedup['demo'] = clean_demographic_data(hepb_data_dedup['demo'])
    
    # Phase 6: Save data
    print(f"\n{'='*80}")
    print("PHASE 6: SAVING EXTRACTED DATA")
    print(f"{'='*80}")
    
    for key, df in hepb_data_dedup.items():
        if df is not None:
            output_path = os.path.join(config.OUTPUT_DIR, f'hepb_{key}.csv')
            df.to_csv(output_path, index=False)
            print(f"✓ Saved: {output_path} ({len(df):,} records)")
    
    # Phase 7: Generate summary
    print(f"\n{'='*80}")
    print("PHASE 7: GENERATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    generate_summary(hepb_data_dedup)
    
    print("\n" + "✅"*40)
    print("EXTRACTION COMPLETE!")
    print("✅"*40)
    
    return hepb_data_dedup


# ============================================================================
# DEDUPLICATION AND CLEANING
# ============================================================================

def deduplicate_cases(hepb_data):
    """Remove duplicate cases (keep most recent version)"""
    if hepb_data['demo'] is None:
        print("No demo data available")
        return hepb_data
    
    demo_df = hepb_data['demo'].copy()
    
    initial_cases = demo_df['PRIMARYID'].nunique()
    print(f"Initial unique cases: {initial_cases:,}")
    
    if 'CASEVERSION' in demo_df.columns:
        demo_df['CASEVERSION'] = pd.to_numeric(demo_df['CASEVERSION'], errors='coerce')
        demo_df = demo_df.sort_values(['CASEID', 'CASEVERSION'], ascending=[True, False])
        demo_df_dedup = demo_df.drop_duplicates(subset=['CASEID'], keep='first')
    else:
        demo_df_dedup = demo_df.drop_duplicates(subset=['PRIMARYID'], keep='first')
    
    final_cases = demo_df_dedup['PRIMARYID'].nunique()
    print(f"Final unique cases: {final_cases:,}")
    print(f"Removed duplicates: {initial_cases - final_cases:,}")
    
    valid_primaryids = set(demo_df_dedup['PRIMARYID'].unique())
    
    filtered_data = {}
    for key, df in hepb_data.items():
        if df is not None:
            if key == 'demo':
                filtered_data[key] = demo_df_dedup
            else:
                filtered_data[key] = df[df['PRIMARYID'].isin(valid_primaryids)].copy()
        else:
            filtered_data[key] = None
    
    return filtered_data


def clean_demographic_data(demo_df):
    """Clean and standardize demographic data"""
    df = demo_df.copy()
    
    # Age cleaning
    if 'AGE' in df.columns:
        df['AGE'] = pd.to_numeric(df['AGE'], errors='coerce')
        
        if 'AGE_COD' in df.columns:
            df['AGE_YEARS'] = df['AGE'].copy()
            df.loc[df['AGE_COD'] == 'DEC', 'AGE_YEARS'] = df['AGE'] * 10
            df.loc[df['AGE_COD'] == 'MON', 'AGE_YEARS'] = df['AGE'] / 12
            df.loc[df['AGE_COD'] == 'WK', 'AGE_YEARS'] = df['AGE'] / 52
            df.loc[df['AGE_COD'] == 'DY', 'AGE_YEARS'] = df['AGE'] / 365
            df.loc[df['AGE_COD'] == 'HR', 'AGE_YEARS'] = df['AGE'] / 8760
            df.loc[(df['AGE_YEARS'] < 0) | (df['AGE_YEARS'] > 120), 'AGE_YEARS'] = np.nan
            
            print(f"✓ Valid ages: {df['AGE_YEARS'].notna().sum():,}")
    
    # Sex cleaning
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].str.upper().str.strip()
        print(f"✓ Sex distribution:")
        for sex, count in df['SEX'].value_counts().items():
            print(f"  {sex}: {count:,}")
    
    # Event date
    if 'EVENT_DT' in df.columns:
        df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DT'], format='%Y%m%d', errors='coerce')
        print(f"✓ Valid dates: {df['EVENT_DATE'].notna().sum():,}")
    
    return df


def generate_summary(hepb_data):
    """Generate summary report"""
    report = []
    report.append("\n" + "="*80)
    report.append("HEPATITIS B EXTRACTION SUMMARY")
    report.append("="*80 + "\n")
    
    if hepb_data['demo'] is not None:
        demo_df = hepb_data['demo']
        report.append(f"Total Cases: {len(demo_df):,}\n")
        
        if 'AGE_YEARS' in demo_df.columns:
            age_valid = demo_df['AGE_YEARS'].dropna()
            if len(age_valid) > 0:
                report.append(f"Age Range: {age_valid.min():.0f} - {age_valid.max():.0f} years")
                report.append(f"Mean Age: {age_valid.mean():.1f} years\n")
        
        if 'SEX' in demo_df.columns:
            report.append("Sex Distribution:")
            for sex, count in demo_df['SEX'].value_counts().items():
                report.append(f"  {sex}: {count:,} ({count/len(demo_df)*100:.1f}%)")
            report.append("")
        
        if 'REPORTER_COUNTRY' in demo_df.columns:
            report.append("Top 10 Reporter Countries:")
            for country, count in demo_df['REPORTER_COUNTRY'].value_counts().head(10).items():
                report.append(f"  {country}: {count:,}")
            report.append("")
    
    if hepb_data['drug'] is not None:
        drug_df = hepb_data['drug']
        report.append(f"Total Drug Records: {len(drug_df):,}")
        report.append(f"Unique Drugs: {drug_df['DRUGNAME'].nunique():,}")
        report.append("\nTop 20 Drugs:")
        for drug, count in drug_df['DRUGNAME'].value_counts().head(20).items():
            report.append(f"  {drug}: {count:,}")
        report.append("")
    
    if hepb_data['reac'] is not None:
        reac_df = hepb_data['reac']
        report.append(f"Total Reaction Records: {len(reac_df):,}")
        report.append(f"Unique Reactions: {reac_df['PT'].nunique():,}")
        report.append("\nTop 20 Adverse Reactions:")
        for reaction, count in reac_df['PT'].value_counts().head(20).items():
            report.append(f"  {reaction}: {count:,}")
        report.append("")
    
    if hepb_data['outc'] is not None:
        outc_df = hepb_data['outc']
        report.append("Outcome Distribution:")
        outcome_codes = {
            'DE': 'Death',
            'LT': 'Life-Threatening',
            'HO': 'Hospitalization',
            'DS': 'Disability',
            'CA': 'Congenital Anomaly',
            'RI': 'Required Intervention',
            'OT': 'Other'
        }
        outcome_counts = outc_df['OUTC_COD'].value_counts()
        for code, count in outcome_counts.items():
            outcome_name = outcome_codes.get(code, code)
            report.append(f"  {outcome_name} ({code}): {count:,}")
        report.append("")
    
    report.append("="*80)
    
    summary_text = "\n".join(report)
    print(summary_text)
    
    summary_path = os.path.join(config.OUTPUT_DIR, 'extraction_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\n✓ Summary report saved: {summary_path}")


# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    hepb_data = run_extraction_pipeline()