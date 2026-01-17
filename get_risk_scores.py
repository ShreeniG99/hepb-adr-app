import pandas as pd

# Load signals
signals_df = pd.read_csv(r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\phase2_analysis\all_signals.csv")

# Calculate mean ROR for each drug
drug_risk_scores = signals_df.groupby('drug')['ror'].mean().to_dict()

with open("drug_scores.txt", "w") as f:
    for drug, score in drug_risk_scores.items():
        f.write(f"'{drug}': {score:.4f},\n")
