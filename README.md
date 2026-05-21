# Hepatitis B ADR Prediction System
**AI-Powered Clinical Decision Support for Drug Safety**

A machine learning system that predicts the risk of Adverse Drug Reactions (ADRs) in Hepatitis B patients *before* treatment, built on real-world FDA pharmacovigilance data.

Click this link to access a pdf file that explains everything about building this project from scratch:
https://drive.google.com/file/d/1IRJ1SZtnxPPd-egwXX_egbKXTkngbwtN/view?usp=sharing


## The Problem

Hepatitis B antiviral drugs cause unpredictable ADRs across 5+ major organ systems, with an incidence rate between 32.6% and 68.3%. Despite this, no existing ML-based prediction tools existed for this specific drug class — clinicians had no data-driven way to assess risk before administering treatment.

---

## What This System Does

Given a patient's demographics, clinical profile, and intended antiviral drug, the system predicts the probability of 5 ADR categories:

- Hepatotoxicity
- Nephrotoxicity
- Bone / Tooth damage
- Hematologic effects
- Neurologic effects

It then surfaces organ-specific risk scores and clinical monitoring recommendations through an interactive web interface.

---

## Dataset

- **Source:** FDA FAERS (Adverse Event Reporting System) — publicly available
- **Period:** 2020–2025 (quarterly updates)
- **Size:** 1,57,079 Hepatitis B patient ADR records
- **Files used:** Demographics (DEMO), Drug exposures (DRUG), Adverse reactions (REAC), Indications (INDI), Outcomes (OUTC)

---

## Pipeline Overview

```
Phase 1: Data Extraction
  └── Downloaded all FAERS quarterly zip files (2020–25)
  └── Filtered, concatenated, deduplicated HBV cases

Phase 2: Data Analysis
  └── Stratified patients into 4 cohorts
  └── Signal detection using ROR, PRR, IC metrics
  └── Volcano plot for cohort-level ADR enrichment

Phase 3A: Feature Engineering
  └── 26-feature matrix → expanded to 56 features with DDI scoring
  └── Demographics + Clinical context + Drug exposure features

Phase 3B: Model Training
  └── Trained: LightGBM, Random Forest, XGBoost,
              Logistic Regression, Stacking Ensemble
  └── Train: 2020–22 | Test: 2025 | Validation: 2023–24

Phase 4: Validation + DDI Risk Scoring
  └── Temporal validation on held-out 2023–24 data
  └── DDI scoring added ROR-derived drug interaction features

Phase 5: Deployment
  └── Streamlit web app for real-time clinical use
```

---

## Patient Cohorts

| Cohort | Share |
|---|---|
| HBV Monoinfection | 72% |
| HIV/HBV Coinfection | 18% |
| HBV + Autoimmune | 7% |
| HBV Other | 3% |

---

## Feature Matrix (56 features)

**Demographic:** Age, Sex, Reporter type (MD / lawyer / consumer)

**Clinical:** Patient cohort, Outcome severity, Severity score (1–5)

**Drug Exposure:** Drug class (6 antivirals), drug-specific ROR risk scores, polypharmacy level, concomitant drug count

**DDI Features (added in Phase 4):** ROR-derived risk score per antiviral × ADR pair (6 drugs × 5 ADRs = 30 interaction features), concomitant risk flags, DDI co-medication counts

---

## Model Results

| Model | Recall | Precision | AUC | F1 | Train Time |
|---|---|---|---|---|---|
| **LightGBM** | **89.0%** | 33.1% | **0.925** | 0.410 | 6.98s |
| Random Forest | 88.4% | 33.1% | 0.922 | 0.409 | 25.96s |
| Logistic Regression | 88.3% | 29.7% | 0.901 | 0.379 | 2.08s |
| Stacking Ensemble | 87.3% | 33.6% | 0.923 | 0.413 | 56.81s |
| XGBoost | 47.2% | 63.0% | 0.926 | 0.488 | 12.86s |

**Why LightGBM?** Highest recall (89%), best AUC (0.925), and 4× faster than the Stacking Ensemble. In a clinical screening context, missing a real ADR (false negative) is more dangerous than a false alarm — so recall was prioritized over precision.

> XGBoost had the highest precision (63%) but only 47% recall — it misses 77% of real ADR cases, which is clinically unacceptable for a screening tool.

---

## Per-ADR Performance (LightGBM)

| ADR Category | Recall | AUC |
|---|---|---|
| Bone / Tooth | 95.8% | 0.977 |
| Nephrotoxicity | 87.7% | 0.954 |
| Neurologic | 89.5% | 0.863 |
| Hematologic | 86.3% | 0.910 |
| Hepatotoxicity | 85.9% | 0.927 |

---

## Temporal Validation

| ADR | Train AUC | Validation AUC | Recall | Verdict |
|---|---|---|---|---|
| Bone / Tooth | 0.975 | 0.973 | 94.7% | Generalizes |
| Nephrotoxicity | 0.952 | 0.951 | 83.9% | Generalizes |
| Hematologic | 0.905 | 0.832 | 39.2% | Moderate drop |
| Neurologic | 0.859 | 0.800 | 44.4% | Moderate drop |
| Hepatotoxicity | 0.919 | 0.850 | 31.1% | Limitation |

**Note:** Hepatotoxicity recall drops to 31% on 2023–24 data due to high class imbalance (685 positives in 54,946 test cases) and evolving reporting patterns in FAERS.

DDI risk scoring improved mean AUC by **+8.5%** by adding co-medication interaction features to the feature set.

---

## Web Application

Built with Streamlit. Takes patient inputs and returns real-time ADR risk predictions.

**Inputs:**
- Patient age and sex
- Clinical cohort (mono / coinfection / autoimmune / other)
- Reporter type
- Antiviral drug selection (Tenofovir, Entecavir, Lamivudine, Adefovir, Telbivudine, Interferon)
- Polypharmacy level and co-medications

**Outputs:**
- Organ-specific ADR risk scores (%)
- Comparative risk analysis across ADR categories
- Clinical monitoring recommendations

**Example use case:** A 62-year-old HIV/HBV-coinfected male already on 5 medications — the clinician selects Tenofovir and gets risk scores across all 5 ADR categories before administration.

---

## Tech Stack

- **Language:** Python
- **ML:** LightGBM, Scikit-Learn, XGBoost, Random Forest
- **Interpretability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

## Team

| Name | Roll No |
|---|---|
| Harithra S | CB.AI.U4AIM24013 |
| Tharika N S | CB.AI.U4AIM24027 |
| Thanushika Sri R A | CB.AI.U4AIM24034 |
| Shreenidhi G | CB.AI.U4AIM24040 |

**Course:** 24AIM211 — Machine Learning for Cheminformatics & Bioinformatics
**Institution:** Amrita Vishwa Vidyapeetham

---

## Signal Detection Methodology

Three pharmacovigilance metrics were used to verify drug-ADR signals against FDA/EMA gold standards:

- **ROR (Reporting Odds Ratio):** Odds of ADR in drug users vs. non-users. Signal detected when ROR ≥ 2.
- **PRR (Proportional Reporting Ratio):** Proportion of ADR reports for a drug vs. all drugs. Signal detected when PRR ≥ 2.
- **IC (Information Component):** Mutual information between drug and ADR. Signal detected when IC > 0.

---

## Limitations

- Hepatotoxicity generalization is weak on temporal holdout — likely due to evolving FAERS reporting patterns and severe class imbalance
- FAERS is a passive surveillance database; reporting bias exists (under-reporting, duplicate entries)
- Model trained on reported cases only — does not account for unreported ADRs
- DDI scoring is ROR-derived, not mechanistic — does not model pharmacokinetic interactions directly
