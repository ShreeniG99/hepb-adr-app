"""
drug_adr_risk_features.py
==========================
Replace polypharmacy_count with drug-specific ADR risk scores.

For each patient, for each ADR target, we encode:
    drug_risk_<adr> = ROR of their specific HBV antiviral for that ADR
                      (from Phase 2 signal detection results)

This closes the loop: Phase 2 pharmacovigilance signals → Phase 3 ML features.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR       = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis"
SIGNALS_PATH   = os.path.join(BASE_DIR, "phase2_analysis",  "all_signals.csv")
FEATURES_PATH  = os.path.join(BASE_DIR, "phase3_features",  "feature_matrix.parquet")
DRUG_PATH      = os.path.join(BASE_DIR, "hepatitis_b_data", "hepb_drug.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "drug_risk_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map ADR target column → ADR terms in your signals CSV
# The right-hand list should loosely match values in the 'adr' column of all_signals.csv
# These are case-insensitive partial matches
ADR_TARGET_MAP = {
    'adr_hepatotoxicity':        ['hepatotoxicity', 'hepatic failure', 'hepatic injury',
                                   'liver disorder', 'alanine aminotransferase increased',
                                   'hepatitis', 'hepatic function abnormal'],
    'adr_nephrotoxicity':        ['renal failure', 'acute kidney injury', 'nephrotoxicity',
                                   'blood creatinine increased', 'renal impairment',
                                   'renal tubular disorder', 'fanconi syndrome'],
    'adr_bone_density_decreas':  ['bone density decreased', 'osteoporosis', 'fracture',
                                   'osteomalacia', 'hypophosphataemia', 'bone disorder'],
    'adr_haematologic_toxicity': ['anaemia', 'thrombocytopenia', 'neutropenia',
                                   'pancytopenia', 'leukopenia', 'haematotoxicity'],
    'adr_tooth_loss':            ['tooth loss', 'tooth disorder', 'dental caries',
                                   'periodontitis', 'gingival disorder', 'halitosis'],
}

HBV_ANTIVIRALS = {
    'ENTECAVIR':   ['ENTECAVIR', 'BARACLUDE'],
    'TENOFOVIR':   ['TENOFOVIR', 'VIREAD', 'VEMLIDY', 'TENOFOVIR DISOPROXIL',
                    'TENOFOVIR ALAFENAMIDE', 'TDF', 'TAF'],
    'LAMIVUDINE':  ['LAMIVUDINE', 'EPIVIR', '3TC'],
    'ADEFOVIR':    ['ADEFOVIR', 'HEPSERA'],
    'TELBIVUDINE': ['TELBIVUDINE', 'TYZEKA'],
    'INTERFERON':  ['PEGINTERFERON ALFA', 'INTERFERON ALFA', 'PEGASYS'],
}

# ============================================================================
# HELPER
# ============================================================================

def load_normalised(path, **kwargs):
    """Load CSV or parquet and force ALL column names to lowercase."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False, dtype=str, **kwargs)
    df.columns = df.columns.str.lower().str.strip()
    return df

def safe_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Columns: {df.columns.tolist()}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("=" * 70)
print("DRUG-SPECIFIC ADR RISK FEATURES")
print("=" * 70)

print("\n[1/5] Loading data...")

# Feature matrix
if os.path.exists(FEATURES_PATH):
    features_df = load_normalised(FEATURES_PATH)
else:
    csv_path = FEATURES_PATH.replace('.parquet', '.csv')
    if not os.path.exists(csv_path):
        sys.exit(f"Feature matrix not found: {FEATURES_PATH}")
    features_df = load_normalised(csv_path)

pid_col = safe_col(features_df, 'primaryid', 'caseid')
features_df[pid_col] = features_df[pid_col].astype(str).str.strip()
if pid_col != 'primaryid':
    features_df = features_df.rename(columns={pid_col: 'primaryid'})
print(f"      Feature matrix: {len(features_df):,} patients")
print(f"      Columns (first 10): {features_df.columns[:10].tolist()}")

# Signals
signals_df = load_normalised(SIGNALS_PATH)
# signals may have mixed case column names — normalise
print(f"      Phase 2 signals: {len(signals_df):,} rows")
print(f"      Signal columns: {signals_df.columns.tolist()}")

# Drug records
drug_df = load_normalised(DRUG_PATH)
pid_drug = safe_col(drug_df, 'primaryid', 'caseid')
drug_df[pid_drug] = drug_df[pid_drug].astype(str).str.strip()
if pid_drug != 'primaryid':
    drug_df = drug_df.rename(columns={pid_drug: 'primaryid'})
print(f"      Drug records: {len(drug_df):,} rows")
print(f"      Drug columns: {drug_df.columns.tolist()}")

# Resolve drug name columns
drugname_col = safe_col(drug_df, 'drugname', 'drug_name')
prodai_col   = 'prod_ai' if 'prod_ai' in drug_df.columns else None

drug_df['_drugname_upper'] = drug_df[drugname_col].fillna('').str.upper()
drug_df['_prodai_upper']   = (
    drug_df[prodai_col].fillna('').str.upper() if prodai_col else pd.Series('', index=drug_df.index)
)

# ============================================================================
# STEP 2: RESOLVE ADR TARGET COLUMNS IN FEATURE MATRIX
# ============================================================================

print("\n[2/5] Resolving ADR target columns...")

# The feature matrix may have slightly different column names
# We try exact match first, then case-insensitive, then partial match
resolved_targets = {}   # canonical_name → actual column in features_df

for canonical, _ in ADR_TARGET_MAP.items():
    # exact
    if canonical in features_df.columns:
        resolved_targets[canonical] = canonical
        continue
    # case-insensitive
    matches = [c for c in features_df.columns if c.lower() == canonical.lower()]
    if matches:
        resolved_targets[canonical] = matches[0]
        continue
    # partial: adr_ prefix match
    stub = canonical.replace('adr_', '')
    matches = [c for c in features_df.columns
               if c.startswith('adr_') and stub[:12] in c]
    if matches:
        resolved_targets[canonical] = matches[0]
        print(f"      Partial match: {canonical} → {matches[0]}")

if not resolved_targets:
    # Fall back: use whatever adr_ columns exist
    auto = [c for c in features_df.columns if c.startswith('adr_')]
    print(f"      No configured targets matched — auto-detected: {auto}")
    resolved_targets = {c: c for c in auto}

print(f"      Resolved {len(resolved_targets)} ADR targets:")
for k, v in resolved_targets.items():
    pos = pd.to_numeric(features_df[v], errors='coerce').sum()
    print(f"        {k} → {v}  ({int(pos):,} positives)")

# ============================================================================
# STEP 3: BUILD ROR LOOKUP FROM PHASE 2 SIGNALS
# ============================================================================

print("\n[3/5] Building ROR lookup from Phase 2 signals...")

# Normalise signals columns
drug_sig_col = safe_col(signals_df, 'drug', 'drug_class')
adr_sig_col  = safe_col(signals_df, 'adr',  'pt')
ror_sig_col  = safe_col(signals_df, 'ror',  'ror_value')

signals_df['_drug_upper'] = signals_df[drug_sig_col].fillna('').str.upper().str.strip()
signals_df['_adr_lower']  = signals_df[adr_sig_col].fillna('').str.lower().str.strip()
signals_df[ror_sig_col]   = pd.to_numeric(signals_df[ror_sig_col], errors='coerce').fillna(1.0)

ror_lookup = {}   # (drug_class, canonical_target) → ROR

for canonical, adr_terms in ADR_TARGET_MAP.items():
    for drug_class in HBV_ANTIVIRALS:
        mask = (
            (signals_df['_drug_upper'] == drug_class) &
            (signals_df['_adr_lower'].apply(
                lambda x: any(term.lower() in x for term in adr_terms)
            ))
        )
        matched = signals_df[mask]
        ror = float(matched[ror_sig_col].max()) if len(matched) > 0 else 1.0
        ror_lookup[(drug_class, canonical)] = ror

print(f"      Lookup entries with signal (ROR > 1): "
      f"{sum(1 for v in ror_lookup.values() if v > 1)}")
for (drug, adr), ror in sorted(ror_lookup.items(), key=lambda x: -x[1]):
    if ror > 2.0:
        print(f"        {drug:12s} | {adr:40s} | ROR = {ror:.1f}")

# ============================================================================
# STEP 4: MAP PATIENTS TO HBV DRUG CLASS
# ============================================================================

print("\n[4/5] Mapping patients to HBV drug class...")

def detect_hbv_class(drugname_upper, prodai_upper):
    combined = drugname_upper + ' ' + prodai_upper
    for cls, patterns in HBV_ANTIVIRALS.items():
        for p in patterns:
            if p in combined:
                return cls
    return None

drug_df['_hbv_class'] = drug_df.apply(
    lambda r: detect_hbv_class(r['_drugname_upper'], r['_prodai_upper']), axis=1
)

patient_hbv = (
    drug_df[drug_df['_hbv_class'].notna()]
    .groupby('primaryid')['_hbv_class']
    .apply(set)
    .reset_index()
)
patient_hbv.columns = ['primaryid', 'hbv_classes']
hbv_map = dict(zip(patient_hbv['primaryid'], patient_hbv['hbv_classes']))
print(f"      Patients with identified HBV drug: {len(hbv_map):,}")

# Drug class distribution
all_classes = []
for s in hbv_map.values():
    all_classes.extend(s)
from collections import Counter
for cls, cnt in Counter(all_classes).most_common():
    print(f"        {cls}: {cnt:,} patients")

# ============================================================================
# STEP 5: COMPUTE RISK FEATURES AND COMPARE MODELS
# ============================================================================

print("\n[5/5] Computing drug-specific ADR risk features...")

risk_rows = []
for pid in features_df['primaryid']:
    row = {'primaryid': pid}
    classes = hbv_map.get(pid, set())
    for canonical in ADR_TARGET_MAP:
        feat = f"drug_risk_{canonical.replace('adr_','')}"
        if classes:
            rors = [ror_lookup.get((cls, canonical), 1.0) for cls in classes]
            row[feat] = max(rors)
        else:
            row[feat] = 1.0
    risk_rows.append(row)

risk_df = pd.DataFrame(risk_rows)
new_feat_cols = [c for c in risk_df.columns if c != 'primaryid']

print(f"\n      New risk features:")
print(f"      {'Feature':<45} {'Mean':>8}  {'Patients w/ signal':>18}")
print(f"      {'-'*45} {'-'*8}  {'-'*18}")
for col in new_feat_cols:
    vals = risk_df[col]
    print(f"      {col:<45} {vals.mean():>8.2f}  {(vals>1).sum():>18,}")

# Merge
features_enhanced = features_df.merge(risk_df, on='primaryid', how='left')
for c in new_feat_cols:
    features_enhanced[c] = features_enhanced[c].fillna(1.0)

# Save
out_path = os.path.join(OUTPUT_DIR, 'feature_matrix_with_drug_risk.parquet')
features_enhanced.to_parquet(out_path, index=False)
print(f"\n      Saved: {out_path}")

# --- Model comparison ---
adr_actual_cols = list(resolved_targets.values())
available = [c for c in adr_actual_cols if c in features_enhanced.columns]

if available:
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (5-fold CV, first ADR target)")
    print("=" * 70)

    target_col = available[0]
    all_exclude = set(['primaryid'] + list(resolved_targets.values()))

    orig_feats = [c for c in features_df.columns if c not in all_exclude]
    new_feats  = orig_feats + new_feat_cols
    # drop polypharmacy_count if present (we're replacing it)
    new_feats  = [c for c in new_feats if 'polypharmacy' not in c]

    n = min(30000, len(features_enhanced))
    samp = features_enhanced.sample(n=n, random_state=42)
    y = pd.to_numeric(samp[target_col], errors='coerce').fillna(0).astype(int)

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=12,
        class_weight='balanced', n_jobs=-1, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X_orig = samp[[c for c in orig_feats if c in samp.columns]].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
    X_new  = samp[[c for c in new_feats  if c in samp.columns]].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)

    auc_orig = cross_val_score(rf, X_orig, y, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    auc_new  = cross_val_score(rf, X_new,  y, cv=cv, scoring='roc_auc', n_jobs=-1).mean()

    label = target_col.replace('adr_', '').replace('_', ' ')
    print(f"\n  ADR: {label}\n")
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Feature Set                      │  AUC (5-CV) │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Polypharmacy count (original)    │  {auc_orig:.3f}      │")
    print(f"  │  Drug-specific ADR risk (new)     │  {auc_new:.3f}      │")
    gain = (auc_new - auc_orig) * 100
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Change                           │  {gain:+.2f}%     │")
    print(f"  └─────────────────────────────────────────────────┘")

    pd.DataFrame({
        'Feature_Set': ['Polypharmacy count', 'Drug-specific ADR risk'],
        'AUC': [round(auc_orig, 3), round(auc_new, 3)],
        'Change_%': [0, round(gain, 2)]
    }).to_csv(os.path.join(OUTPUT_DIR, 'drug_risk_comparison.csv'), index=False)

print("\n✅ drug_adr_risk_features.py complete")