# Clinical Interpretation Report: Hepatitis B ADR Prediction

**Generated:** 2026-01-29 09:01:16

---

## Executive Summary

This report provides SHAP-based explanations for the Hepatitis B Adverse Drug Reaction (ADR) prediction model. SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to predictions, enabling clinical interpretation of risk factors.

## Overall Top 10 Risk Factors

| Rank | Feature | Mean Importance | Direction |
|------|---------|-----------------|----------|
| 2 | Legal Representative Report | 0.6882 | Increases Risk |
| 3 | Composite Severity Score | 0.2560 | Increases Risk |
| 5 | Tenofovir Use | 0.1750 | Increases Risk |
| 8 | HIV/HBV Co-infection | 0.1158 | Increases Risk |
| 9 | Healthcare Provider Report | 0.0892 | Increases Risk |
| 10 | Prior Hospitalization | 0.0863 | Increases Risk |
| 11 | Female Sex | 0.0422 | Increases Risk |
| 15 | Interferon Use | 0.0241 | Increases Risk |
| 16 | Lamivudine Use | 0.0229 | Increases Risk |
| 19 | Entecavir Use | 0.0170 | Increases Risk |

## Top 5 Protective Factors

| Rank | Feature | Mean Importance | Direction |
|------|---------|-----------------|----------|
| 1 | Drug Risk Score (ROR) | 4.0379 | Decreases Risk |
| 2 | Total Concurrent Medications | 0.2223 | Decreases Risk |
| 3 | Patient Age (years) | 0.1350 | Decreases Risk |
| 4 | Consumer Report | 0.1186 | Decreases Risk |
| 5 | Prior Death Outcome | 0.0396 | Decreases Risk |

---

## ADR-Specific Analysis

### Hepatotoxicity (Liver Damage)

**Top 5 Risk Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Legal Representative Report | 0.4737 |
| 2 | Composite Severity Score | 0.3203 |
| 3 | HIV/HBV Co-infection | 0.2260 |
| 4 | Healthcare Provider Report | 0.1247 |
| 5 | Prior Hospitalization | 0.0476 |

**Protective Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Drug Risk Score (ROR) | 4.1689 |
| 2 | Patient Age (years) | 0.2065 |
| 3 | Total Concurrent Medications | 0.1759 |


### Nephrotoxicity (Kidney Damage)

**Top 5 Risk Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Legal Representative Report | 0.9634 |
| 2 | Composite Severity Score | 0.4316 |
| 3 | Prior Hospitalization | 0.1168 |
| 4 | Female Sex | 0.0893 |
| 5 | Healthcare Provider Report | 0.0351 |

**Protective Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Drug Risk Score (ROR) | 4.2290 |
| 2 | Tenofovir Use | 0.1094 |
| 3 | HIV/HBV Co-infection | 0.0477 |


### Bone/Dental Complications

**Top 5 Risk Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Legal Representative Report | 1.3670 |
| 2 | Tenofovir Use | 0.6115 |
| 3 | Patient Age (years) | 0.1311 |
| 4 | Interferon Use | 0.0413 |
| 5 | Entecavir Use | 0.0356 |

**Protective Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Drug Risk Score (ROR) | 3.3791 |
| 2 | Consumer Report | 0.3455 |
| 3 | HIV/HBV Co-infection | 0.0947 |


### Hematologic Toxicity (Blood Disorders)

**Top 5 Risk Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Composite Severity Score | 0.1960 |
| 2 | Healthcare Provider Report | 0.1793 |
| 3 | Prior Hospitalization | 0.1745 |
| 4 | HIV/HBV Co-infection | 0.1667 |
| 5 | Legal Representative Report | 0.1423 |

**Protective Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Drug Risk Score (ROR) | 4.1730 |
| 2 | Total Concurrent Medications | 0.3851 |
| 3 | Consumer Report | 0.0573 |


### Neurologic Toxicity (Nerve Damage)

**Top 5 Risk Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Legal Representative Report | 0.4945 |
| 2 | Patient Age (years) | 0.1152 |
| 3 | Healthcare Provider Report | 0.0813 |
| 4 | Prior Hospitalization | 0.0620 |
| 5 | Lamivudine Use | 0.0278 |

**Protective Factors:**

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | Drug Risk Score (ROR) | 4.2393 |
| 2 | Prior Death Outcome | 0.1289 |
| 3 | HIV/HBV Co-infection | 0.0438 |


---

## Methodology

- **SHAP Algorithm:** TreeExplainer for tree-based ensemble models
- **Sample Size:** 5,000 patients from test set
- **Interpretation:** Positive SHAP values indicate features that increase ADR risk; negative values indicate protective factors
- **Feature Importance:** Calculated as mean absolute SHAP value across all samples

## Limitations

1. SHAP values represent average feature contributions and may vary for individual patients
2. Correlations between features may affect individual SHAP value interpretations
3. This analysis is based on observational FAERS data and should not be used for individual clinical decisions without additional validation
4. Drug-specific risks require consideration of indication, dose, and duration

---

*Report generated by SHAP Explainability Pipeline*
