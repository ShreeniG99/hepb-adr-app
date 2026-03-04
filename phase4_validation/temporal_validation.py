"""
temporal_validation.py
=======================
Temporal Validation for HBV ADR Prediction Model
Train on 2020-2022, Test on 2023-2024
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION — update paths if needed
# ============================================================================

BASE_DIR      = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis"
FEATURES_PATH = os.path.join(BASE_DIR, "phase3_features", "feature_matrix.parquet")
DEMO_PATH     = os.path.join(BASE_DIR, "hepatitis_b_data", "hepb_demo.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "phase4_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_YEARS = [2020, 2021, 2022]
TEST_YEARS  = [2023, 2024]

# ============================================================================
# HELPER: load any file and immediately normalise all columns to lowercase
# ============================================================================

def load_normalised(path, **kwargs):
    """Load CSV or parquet, force ALL column names to lowercase."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False, dtype=str, **kwargs)
    df.columns = df.columns.str.lower().str.strip()
    return df

def safe_col(df, *candidates):
    """Return the first candidate column name that actually exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")

# ============================================================================
# STEP 1: LOAD FEATURE MATRIX
# ============================================================================

print("=" * 70)
print("TEMPORAL VALIDATION — HBV ADR PREDICTION")
print("=" * 70)

print("\n[1/5] Loading feature matrix...")
if os.path.exists(FEATURES_PATH):
    features_df = load_normalised(FEATURES_PATH)
    print(f"      Loaded parquet: {len(features_df):,} rows")
else:
    csv_path = FEATURES_PATH.replace('.parquet', '.csv')
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: Feature matrix not found at:\n  {FEATURES_PATH}\n  {csv_path}")
    features_df = load_normalised(csv_path)
    print(f"      Loaded CSV: {len(features_df):,} rows")

print(f"      Columns (first 10): {features_df.columns[:10].tolist()}")

# Resolve primaryid column
pid_col = safe_col(features_df, 'primaryid', 'caseid')
features_df[pid_col] = features_df[pid_col].astype(str).str.strip()
if pid_col != 'primaryid':
    features_df = features_df.rename(columns={pid_col: 'primaryid'})

# Auto-detect ADR target columns
adr_targets = [c for c in features_df.columns if c.startswith('adr_')]
if not adr_targets:
    sys.exit("ERROR: No columns starting with 'adr_' found in feature matrix.\n"
             f"All columns: {features_df.columns.tolist()}")
print(f"\n      Auto-detected {len(adr_targets)} ADR targets:")
for t in adr_targets:
    pos = pd.to_numeric(features_df[t], errors='coerce').sum()
    pct = 100 * pos / len(features_df)
    print(f"        {t}: {int(pos):,} positives ({pct:.1f}%)")

# ============================================================================
# STEP 2: ATTACH REPORT YEAR FROM DEMO FILE
# ============================================================================

print("\n[2/5] Attaching report year from demo file...")
demo_df = load_normalised(DEMO_PATH)
demo_pid = safe_col(demo_df, 'primaryid', 'caseid')
demo_df[demo_pid] = demo_df[demo_pid].astype(str).str.strip()
if demo_pid != 'primaryid':
    demo_df = demo_df.rename(columns={demo_pid: 'primaryid'})

# Derive year — try year_quarter first, then event_dt, then i_f_code year
if 'year_quarter' in demo_df.columns:
    demo_df['report_year'] = demo_df['year_quarter'].str.extract(r'(\d{4})').astype(float)
    print("      Year source: year_quarter column")
elif 'event_dt' in demo_df.columns:
    demo_df['report_year'] = demo_df['event_dt'].astype(str).str[:4].apply(
        pd.to_numeric, errors='coerce'
    )
    print("      Year source: event_dt column")
elif 'fda_dt' in demo_df.columns:
    demo_df['report_year'] = demo_df['fda_dt'].astype(str).str[:4].apply(
        pd.to_numeric, errors='coerce'
    )
    print("      Year source: fda_dt column")
else:
    sys.exit("ERROR: Cannot find year column in demo file.\n"
             f"Demo columns: {demo_df.columns.tolist()}")

year_map = demo_df[['primaryid', 'report_year']].drop_duplicates('primaryid')
features_df = features_df.merge(year_map, on='primaryid', how='left')

valid_years = features_df['report_year'].dropna()
print(f"      Cases with year: {len(valid_years):,} / {len(features_df):,}")
print(f"      Year distribution:")
for yr, cnt in valid_years.value_counts().sort_index().items():
    tag = "TRAIN" if int(yr) in TRAIN_YEARS else "TEST "
    print(f"        [{tag}] {int(yr)}: {cnt:,} cases")

# ============================================================================
# STEP 3: TEMPORAL SPLIT
# ============================================================================

print("\n[3/5] Splitting data temporally...")

train_mask = features_df['report_year'].isin(TRAIN_YEARS)
test_mask  = features_df['report_year'].isin(TEST_YEARS)

exclude_cols  = set(['primaryid', 'report_year'] + adr_targets)
feature_cols  = [c for c in features_df.columns if c not in exclude_cols]

train_df = features_df[train_mask].copy()
test_df  = features_df[test_mask].copy()

print(f"      Training set (2020-2022): {len(train_df):,} cases")
print(f"      Test set     (2023-2024): {len(test_df):,} cases")

if len(train_df) == 0 or len(test_df) == 0:
    print("\n  WARNING: One split is empty.")
    print("  Checking year column values in features_df:")
    print(features_df['report_year'].value_counts())
    sys.exit("Cannot proceed with empty split.")

X_train = train_df[feature_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
X_test  = test_df[feature_cols].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
y_train = train_df[adr_targets].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
y_test  = test_df[adr_targets].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# ============================================================================
# STEP 4: TRAIN MODEL
# ============================================================================

print("\n[4/5] Training Random Forest on 2020-2022...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15,
    class_weight='balanced', n_jobs=-1, random_state=42
)
model = MultiOutputClassifier(rf, n_jobs=-1)
model.fit(X_train, y_train)
print("      Training complete.")

# ============================================================================
# STEP 5: EVALUATE ON 2023-2024
# ============================================================================

print("\n[5/5] Evaluating on 2023-2024 (unseen future data)...")

y_pred       = model.predict(X_test)
y_pred_proba = np.array([
    est.predict_proba(X_test)[:, 1] for est in model.estimators_
]).T

print("\n" + "=" * 70)
print("TEMPORAL VALIDATION RESULTS")
print("=" * 70)

results = []
for i, adr in enumerate(adr_targets):
    label     = adr.replace('adr_', '').replace('_', ' ')
    y_true_i  = y_test.iloc[:, i].values
    y_pred_i  = y_pred[:, i]
    y_prob_i  = y_pred_proba[:, i]

    if y_true_i.sum() == 0:
        print(f"\n  {label}: no positives in test set — skipped")
        continue

    auc  = roc_auc_score(y_true_i, y_prob_i)
    rec  = recall_score(y_true_i, y_pred_i, zero_division=0)
    prec = precision_score(y_true_i, y_pred_i, zero_division=0)
    f1   = f1_score(y_true_i, y_pred_i, zero_division=0)

    print(f"\n  ► {label}")
    print(f"      AUC:            {auc:.3f}")
    print(f"      Recall:         {rec:.3f}  ← sensitivity (clinical catch rate)")
    print(f"      Precision:      {prec:.3f}")
    print(f"      F1:             {f1:.3f}")
    print(f"      Test positives: {y_true_i.sum():,} / {len(y_true_i):,}")

    results.append({
        'ADR': label, 'AUC': round(auc, 3),
        'Recall': round(rec, 3), 'Precision': round(prec, 3), 'F1': round(f1, 3),
        'Test_positives': int(y_true_i.sum()), 'Test_total': len(y_true_i),
        'Train_size': len(train_df), 'Test_size': len(test_df),
    })

if results:
    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print(f"  Mean AUC  (temporal): {results_df['AUC'].mean():.3f}")
    print(f"  Mean Recall:          {results_df['Recall'].mean():.3f}")
    out = os.path.join(OUTPUT_DIR, 'temporal_validation_results.csv')
    results_df.to_csv(out, index=False)
    print(f"\n  Results saved: {out}")

print("""
NOTE FOR PRESENTATION:
  Temporal validation is stricter than random cross-validation because
  the model is tested on future data it has never seen — exactly how
  it would work in a real clinical deployment. A small AUC drop vs.
  cross-validation is expected and is a sign of honest evaluation.
""")