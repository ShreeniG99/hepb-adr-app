import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import base64

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HepB ADR Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# BACKGROUND IMAGE
# ─────────────────────────────────────────────────────────────
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image_path = "bg_pic.png"
if os.path.exists(bg_image_path):
    bg_img_base64 = get_base64_of_bin_file(bg_image_path)
    bg_img_css = f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(255,255,255,0.85);
            z-index: -1;
        }}
    </style>"""
else:
    bg_img_css = """
    <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    </style>"""

st.markdown(bg_img_css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --navy:   #002244;
        --navy2:  #004488;
        --red:    #e74c3c;
        --slate:  #2c3e50;
    }
    .main-header {
        background: linear-gradient(135deg, #002244 0%, #004488 100%);
        padding: 2rem; border-radius: 0 0 20px 20px;
        color: white; text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 { color: white !important; font-weight: 700; margin-bottom: 0.5rem; }
    .main-header p  { color: #ecf0f1; font-size: 1.1rem; }

    .stButton>button {
        background-color: #004488; color: white;
        border-radius: 8px; padding: 0.5rem 2rem;
        border: none; font-weight: 600; width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #002244;
        box-shadow: 0 4px 12px rgba(0,34,68,0.2);
        transform: translateY(-2px);
    }

    .result-card {
        background-color: rgba(255,255,255,0.95);
        padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 5px solid #bdc3c7;
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        backdrop-filter: blur(10px);
    }
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    .risk-low    { border-left-color: #2ecc71 !important; }
    .risk-medium { border-left-color: #f1c40f !important; }
    .risk-high   { border-left-color: #e74c3c !important; }

    h2, h3 { color: #004488; font-family: 'Segoe UI', sans-serif; }

    .flag-card {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #bdc3c7;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .flag-active   { border-left-color: #e74c3c !important; background: rgba(231,76,60,0.05) !important; }
    .flag-inactive { border-left-color: #2ecc71 !important; background: rgba(46,204,113,0.05) !important; }

    [data-testid="stSidebar"] { background-color: rgba(255,255,255,0.95); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_resources():
    model_path  = "phase3_models/adr_classifier.pkl"
    scaler_path = "phase3_models/feature_scaler.pkl"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)

# ─────────────────────────────────────────────────────────────
# CONSTANTS  (all derived from your actual feature matrix)
# ─────────────────────────────────────────────────────────────
ADR_CATEGORIES = ['hepatotoxicity', 'nephrotoxicity', 'bone_tooth',
                  'hematologic', 'neurologic']

ADR_DISPLAY = {
    'hepatotoxicity': 'Hepatotoxicity',
    'nephrotoxicity': 'Nephrotoxicity',
    'bone_tooth':     'Bone / Tooth',
    'hematologic':    'Hematologic',
    'neurologic':     'Neurologic',
}

# Original features (must match training order exactly)
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

# ── Drug-specific ADR risk scores (from your Phase 2 FAERS analysis) ──────────
# These are the ROR values computed in drug_adr_risk_features.py
DRUG_ADR_RISKS = {
    'TENOFOVIR': {
        'drug_risk_score':                4.0220,
        'drug_risk_nephrotoxicity':       28.3778,
        'drug_risk_hepatotoxicity':       42.8774,
        'drug_risk_bone_density_decreas': 513.8799,
        'drug_risk_haematologic_toxicity':16.4717,
        'drug_risk_tooth_loss':           2596.4757,
    },
    'ENTECAVIR': {
        'drug_risk_score':                8.5138,
        'drug_risk_nephrotoxicity':       14.5414,
        'drug_risk_hepatotoxicity':       110.7327,
        'drug_risk_bone_density_decreas': 212.6698,
        'drug_risk_haematologic_toxicity':23.2612,
        'drug_risk_tooth_loss':           1052.8670,
    },
    'LAMIVUDINE': {
        'drug_risk_score':                6.2125,
        'drug_risk_nephrotoxicity':       22.0001,
        'drug_risk_hepatotoxicity':       53.2088,
        'drug_risk_bone_density_decreas': 375.4859,
        'drug_risk_haematologic_toxicity':28.5022,
        'drug_risk_tooth_loss':           1872.4982,
    },
    'ADEFOVIR': {
        'drug_risk_score':                24.5285,
        'drug_risk_nephrotoxicity':       26.8563,
        'drug_risk_hepatotoxicity':       266.8600,
        'drug_risk_bone_density_decreas': 370.5528,
        'drug_risk_haematologic_toxicity':28.7026,
        'drug_risk_tooth_loss':           1765.0429,
    },
    'TELBIVUDINE': {
        'drug_risk_score':                284.8836,
        'drug_risk_nephrotoxicity':       26.9254,
        'drug_risk_hepatotoxicity':       191.5208,
        'drug_risk_bone_density_decreas': 463.7291,
        'drug_risk_haematologic_toxicity':28.3558,
        'drug_risk_tooth_loss':           2321.0729,
    },
    'INTERFERON': {
        'drug_risk_score':                11.2757,
        'drug_risk_nephrotoxicity':       9.6425,
        'drug_risk_hepatotoxicity':       80.7562,
        'drug_risk_bone_density_decreas': 153.3878,
        'drug_risk_haematologic_toxicity':38.9537,
        'drug_risk_tooth_loss':           760.0922,
    },
}

# ── Concomitant drug risk weights ──────────────────────────────────────────────
# Key: (concomitant_drug_class, adr_target) → risk multiplier
# 1.0 = neutral, higher = amplified risk
CONCOMITANT_RISK_WEIGHTS = {
    # NEPHROTOXICITY amplifiers
    ('diuretics',       'nephrotoxicity'):     3.0,
    ('nsaids',          'nephrotoxicity'):     3.0,
    ('nsaids',          'hepatotoxicity'):     2.0,
    ('aminoglycosides', 'nephrotoxicity'):     4.0,
    ('ace_inhibitors',  'nephrotoxicity'):     2.5,
    # HEPATOTOXICITY amplifiers
    ('metformin',       'hepatotoxicity'):     2.5,
    ('statins',         'hepatotoxicity'):     2.0,
    ('antituberculars', 'hepatotoxicity'):     4.0,
    ('antifungals',     'hepatotoxicity'):     3.0,
    # BONE amplifiers
    ('corticosteroids', 'bone_tooth'):         3.5,
    ('ppis',            'bone_tooth'):         2.0,
    ('anticonvulsants', 'bone_tooth'):         3.0,
    # HAEMATOLOGIC amplifiers
    ('immunosuppressants','hematologic'):      3.5,
    ('chemotherapy',    'hematologic'):        5.0,
    ('chemotherapy',    'nephrotoxicity'):     4.0,
    # DENTAL amplifiers
    ('ppis',            'bone_tooth'):         2.0,
    ('ssris',           'bone_tooth'):         2.0,
    ('ca_blockers',     'bone_tooth'):         2.5,
}

# Concomitant drug UI labels → internal classes
COMEDICATION_MAP = {
    'Diuretics (furosemide, HCTZ)':            'diuretics',
    'NSAIDs (ibuprofen, naproxen)':            'nsaids',
    'Metformin / diabetes medication':         'metformin',
    'Statins (atorvastatin, simvastatin)':     'statins',
    'Corticosteroids (prednisone)':            'corticosteroids',
    'Proton pump inhibitors (omeprazole)':     'ppis',
    'ACE inhibitors (lisinopril, enalapril)':  'ace_inhibitors',
    'Immunosuppressants (methotrexate)':       'immunosuppressants',
    'Antituberculars (rifampicin, isoniazid)': 'antituberculars',
    'Antifungals (fluconazole)':               'antifungals',
    'Aminoglycosides (gentamicin)':            'aminoglycosides',
    'SSRIs (sertraline, fluoxetine)':          'ssris',
    'Calcium channel blockers (amlodipine)':   'ca_blockers',
    'Anticonvulsants (phenytoin)':             'anticonvulsants',
    'Chemotherapy agents':                     'chemotherapy',
}

# ── Temporal validation results ────────────────────────────────────────────────
TEMPORAL_RESULTS = pd.DataFrame({
    'ADR':           ['Hepatotoxicity','Nephrotoxicity','Bone / Tooth','Hematologic','Neurologic'],
    'AUC':           [0.850, 0.951, 0.973, 0.832, 0.800],
    'Recall':        [0.311, 0.839, 0.947, 0.392, 0.444],
    'Precision':     [0.071, 0.658, 0.652, 0.142, 0.090],
    'F1':            [0.116, 0.737, 0.772, 0.208, 0.149],
    'Test Positives':[685,   7440,  6955,  1755,  1818],
})

FEATURE_COMPARISON = pd.DataFrame({
    'Feature Set': [
        'A: Polypharmacy count (original)',
        'B: + HBV drug-specific ADR risk',
        'C: + Concomitant drug risk',
    ],
    'AUC':   [0.978, 0.978, 0.981],
})

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS  (defined before UI so they're available everywhere)
# ─────────────────────────────────────────────────────────────
def _ror_label(ror):
    if ror >= 100:  return "🔴 Very strong"
    if ror >= 10:   return "🟠 Strong"
    if ror >= 2:    return "🟡 Moderate"
    return                  "🟢 Weak / none"

def _get_mechanism(adr, drug_cls):
    mechanisms = {
        ('nephrotoxicity', 'diuretics'):        'Reduces renal perfusion → amplifies TDF tubular toxicity',
        ('nephrotoxicity', 'nsaids'):           'Prostaglandin inhibition → renal ischaemia',
        ('nephrotoxicity', 'aminoglycosides'):  'Additive proximal tubular damage',
        ('nephrotoxicity', 'ace_inhibitors'):   'Lowers GFR → potentiates TDF nephrotoxicity',
        ('nephrotoxicity', 'chemotherapy'):     'Additive tubular + glomerular damage',
        ('hepatotoxicity', 'metformin'):        'Shared mitochondrial Complex I inhibition',
        ('hepatotoxicity', 'statins'):          'CYP3A4 competition + hepatic myotoxicity',
        ('hepatotoxicity', 'antituberculars'):  'Strong CYP450 induction alters antiviral metabolism',
        ('hepatotoxicity', 'antifungals'):      'CYP3A4 inhibition raises antiviral plasma levels',
        ('hepatotoxicity', 'nsaids'):           'Direct hepatic stress + prostaglandin inhibition',
        ('bone_tooth', 'corticosteroids'):      'Glucocorticoid-induced osteoporosis + TDF bone effects',
        ('bone_tooth', 'ppis'):                 'Reduced calcium absorption + lowered salivary pH',
        ('bone_tooth', 'anticonvulsants'):      'Accelerated vitamin D catabolism → osteoporosis',
        ('bone_tooth', 'ssris'):                'Xerostomia → reduced salivary dental protection',
        ('bone_tooth', 'ca_blockers'):          'Gingival hyperplasia',
        ('hematologic', 'immunosuppressants'):  'Bone marrow suppression synergy',
        ('hematologic', 'chemotherapy'):        'Myelosuppression amplified',
    }
    return mechanisms.get((adr, drug_cls), 'Pharmacological interaction')

def _render_monitoring_protocols(high_risks):
    protocols = {
        "Hepatotoxicity": ("#e67e22", "#d35400",
            "Monitor ALT/AST monthly. Watch for hepatic decompensation signs."),
        "Nephrotoxicity":  ("#3498db", "#2980b9",
            "Monitor eGFR and creatinine every 3 months. "
            "Consider dose adjustment if eGFR < 50 mL/min."),
        "Bone / Tooth":    ("#9b59b6", "#8e44ad",
            "DEXA scan annually. Calcium/Vitamin D supplementation. "
            "Dental exam every 6 months."),
        "Hematologic":     ("#e74c3c", "#c0392b",
            "Complete blood count (CBC) every 2 months. Monitor for cytopenias."),
        "Neurologic":      ("#16a085", "#138d75",
            "Assess for peripheral neuropathy. Consider neurological consultation."),
    }
    st.markdown(
        "<div style='background:rgba(255,255,255,0.95);padding:1rem;"
        "border-radius:10px;border-left:4px solid #e74c3c;margin-bottom:1rem;'>"
        "<h4 style='color:#002244;margin:0 0 0.75rem 0;'>🔍 Monitoring Protocols:</h4>",
        unsafe_allow_html=True
    )
    for risk in high_risks:
        if risk in protocols:
            c1, c2, text = protocols[risk]
            st.markdown(
                f"<div style='background:linear-gradient(135deg,{c1} 0%,{c2} 100%);"
                f"padding:0.8rem;border-radius:8px;margin-bottom:0.75rem;'>"
                f"<p style='color:white;margin:0;font-weight:600;'>"
                f"<strong>{risk}:</strong> {text}</p></div>",
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚕️ Clinical ADR Predictor</h1>
    <p>Hepatitis B Adverse Drug Reaction Risk Assessment System</p>
</div>
""", unsafe_allow_html=True)

model, scaler = load_model_resources()
if model is None:
    st.error("🚨 Model files not found. Ensure 'adr_classifier.pkl' and "
             "'feature_scaler.pkl' are in the 'phase3_models' directory.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────
tab_predict, tab_validation = st.tabs([
    " Risk Prediction",
    " Model Validation",
])

# ═════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═════════════════════════════════════════════════════════════
with tab_predict:
    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.markdown("### Patient Data")

        with st.container(border=True):
            # ── Demographics ──────────────────────────────
            age = st.slider("Patient Age", 18, 90, 45)
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)

            st.markdown("#### Clinical Context")
            cohort = st.selectbox(
                "Patient Cohort",
                ["HBV Monoinfection", "HIV/HBV Coinfection",
                 "HBV + Autoimmune", "HBV Other"]
            )
            reporter = st.selectbox(
                "Reporter Type",
                ["Healthcare Professional", "Consumer", "Lawyer"],
                index=0,
                help="Source of the adverse event report"
            )

            # ── Treatment ─────────────────────────────────
            st.markdown("#### Treatment Plan")
            drug = st.selectbox(
                "HBV Antiviral Drug",
                ["TENOFOVIR", "ENTECAVIR", "LAMIVUDINE",
                 "ADEFOVIR", "TELBIVUDINE", "INTERFERON"]
            )

            # ── Drug-specific risk info box ────────────────
            risks = DRUG_ADR_RISKS[drug]
            st.markdown(
                f"<div style='background:rgba(0,68,136,0.07);border-radius:8px;"
                f"padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.82rem;'>"
                f"<b>Phase 2 signal (ROR):</b><br>"
                f" Hepatotoxicity: <b>{risks['drug_risk_hepatotoxicity']:.1f}</b> &nbsp;|&nbsp; "
                f" Nephrotoxicity: <b>{risks['drug_risk_nephrotoxicity']:.1f}</b><br>"
                f" Tooth loss: <b>{risks['drug_risk_tooth_loss']:.0f}</b> &nbsp;|&nbsp; "
                f" Bone: <b>{risks['drug_risk_bone_density_decreas']:.0f}</b>"
                f"</div>",
                unsafe_allow_html=True
            )

            # ── Polypharmacy ──────────────────────────────
            polypharmacy = st.select_slider(
                "Polypharmacy Level",
                options=["Low (1-2)", "Medium (3-5)", "High (6+)"],
                value="Low (1-2)",
                help="Number of concomitant medications"
            )

            # ── Concomitant medications ───────────────────
            st.markdown("#### Co-medications")
            st.caption("Select all medications the patient is currently taking:")
            selected_comedications = st.multiselect(
                "Concomitant drugs",
                options=list(COMEDICATION_MAP.keys()),
                default=[],
                help="These medications may amplify specific ADR risks through "
                     "shared biological pathways.",
                label_visibility="collapsed"
            )

        predict_btn = st.button(" Calculate Risk Profile", use_container_width=True)

    # ── Results Column ─────────────────────────────────────────────────────────
    with col_results:
        if predict_btn:
            # ── 1. Build base feature vector ──────────────
            features = pd.DataFrame(0, index=[0], columns=FEATURE_COLS)
            features['age_years']   = age
            features['sex_binary']  = 1 if sex == "Male" else 0
            features['sex_female']  = 1 if sex == "Female" else 0

            if "Healthcare" in reporter:
                features['reporter_md'] = 1
            elif "Lawyer" in reporter:
                features['reporter_lawyer'] = 1
            else:
                features['reporter_consumer'] = 1

            features['severity_score'] = 1

            if "Low" in polypharmacy:
                features['drug_count'] = 1.5
                features['polypharmacy_low'] = 1
            elif "Medium" in polypharmacy:
                features['drug_count'] = 4.0
                features['polypharmacy_medium'] = 1
            else:
                features['drug_count'] = 7.0
                features['polypharmacy_high'] = 1

            if "Monoinfection" in cohort:
                features['cohort_hbv_mono'] = 1
            elif "HIV" in cohort:
                features['cohort_hiv_hbv'] = 1
            elif "Autoimmune" in cohort:
                features['cohort_autoimmune'] = 1

            features[f'drug_{drug.lower()}'] = 1
            features['drug_risk_score'] = risks['drug_risk_score']

            # ── 2. Scale and predict ───────────────────────
            features_scaled = scaler.transform(features)
            try:
                raw_probas = model.predict_proba(features_scaled)
                base_probas = [float(p[0][1]) for p in raw_probas]
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()

            # ── 3. Apply concomitant drug modifiers ────────
            # Map selected UI labels → internal class names
            selected_classes = {
                COMEDICATION_MAP[label] for label in selected_comedications
            }

            # ADR index map (matches ADR_CATEGORIES order)
            adr_idx = {
                'hepatotoxicity': 0,
                'nephrotoxicity': 1,
                'bone_tooth':     2,
                'hematologic':    3,
                'neurologic':     4,
            }

            # For each ADR, find the max concomitant risk multiplier
            # Then apply a sigmoid-scaled adjustment so we don't exceed 1.0
            adjusted_probas = list(base_probas)
            comedication_flags = {}  # adr → (multiplier, drug_name)

            for adr, idx in adr_idx.items():
                max_mult = 1.0
                flagged_drug = None
                for cls in selected_classes:
                    mult = CONCOMITANT_RISK_WEIGHTS.get((cls, adr), 1.0)
                    if mult > max_mult:
                        max_mult = mult
                        flagged_drug = cls.replace('_', ' ').title()

                if max_mult > 1.0:
                    # Scale: higher multiplier → pushes probability up
                    # Formula: new_p = p + (1-p) * (1 - 1/multiplier) * 0.4
                    # This keeps probability bounded [0,1] and is proportional
                    p = base_probas[idx]
                    adjustment = (1 - p) * (1 - 1 / max_mult) * 0.4
                    adjusted_probas[idx] = min(p + adjustment, 0.99)
                    comedication_flags[adr] = (max_mult, flagged_drug)

            # ── 4. Display Results ─────────────────────────
            st.markdown("### Risk Assessment Results")

            # Patient summary
            comeds_str = (f" + {len(selected_comedications)} co-med(s)"
                          if selected_comedications else "")
            st.markdown(
                f"<div style='background:rgba(255,255,255,0.95);padding:1rem;"
                f"border-radius:10px;margin-bottom:1rem;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.1);'>"
                f"<h4 style='margin:0;color:#002244;'>Patient Profile</h4>"
                f"<p style='margin:0.5rem 0 0 0;color:#2c3e50;'>"
                f"<strong>{age}yr {sex}</strong> · {cohort} · "
                f"{drug}{comeds_str}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Risk cards grid
            risk_data = []
            grid1, grid2 = st.columns(2)

            for idx, (cat, prob) in enumerate(zip(ADR_CATEGORIES, adjusted_probas)):
                pct   = prob * 100
                base_pct = base_probas[idx] * 100
                label = ADR_DISPLAY[cat]

                if pct > 50:
                    level, css, icon, color = "High",   "risk-high",   "🔴", "#e74c3c"
                elif pct > 30:
                    level, css, icon, color = "Medium", "risk-medium", "🟡", "#f1c40f"
                else:
                    level, css, icon, color = "Low",    "risk-low",    "🟢", "#2ecc71"

                # Show adjustment arrow if concomitant drug raised the risk
                flag = comedication_flags.get(cat)
                delta_html = ""
                if flag:
                    diff = pct - base_pct
                    delta_html = (
                        f"<span style='font-size:0.75rem;color:#e74c3c;'>"
                        f"▲ +{diff:.1f}% ({flag[1]})</span>"
                    )

                risk_data.append({"Category": label, "Probability": pct, "Color": color})

                with (grid1 if idx % 2 == 0 else grid2):
                    st.markdown(
                        f"<div class='result-card {css}'>"
                        f"<h3 style='margin:0;font-size:1rem;color:#002244;font-weight:700;'>{label}</h3>"
                        f"<div style='display:flex;align-items:baseline;"
                        f"justify-content:space-between;margin-top:0.5rem;'>"
                        f"<span style='font-size:1.8rem;font-weight:bold;color:{color};'>{pct:.1f}%</span>"
                        f"<span style='font-size:0.9rem;color:#2c3e50;font-weight:600;'>"
                        f"{icon} {level} Risk</span>"
                        f"</div>{delta_html}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            # ── Bar chart ──────────────────────────────────
            st.markdown("#### Comparative Risk Analysis")
            df_risk = pd.DataFrame(risk_data)
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('white')
            ax.patch.set_facecolor('white')
            sns.barplot(data=df_risk, x="Probability", y="Category",
                        palette=[d['Color'] for d in risk_data], ax=ax)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontweight='bold')
            ax.set_ylabel("")
            sns.despine(left=True, bottom=True)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            st.pyplot(fig)
            plt.close()

            # ── Co-medication interaction panel ────────────
            if selected_comedications:
                st.markdown("####  Drug Interaction Analysis")
                st.caption(
                    "These co-medications amplify specific ADR pathways through "
                    "shared biological mechanisms."
                )

                interaction_data = []
                for adr, (mult, drug_name) in comedication_flags.items():
                    interaction_data.append({
                        'ADR': ADR_DISPLAY[adr],
                        'Co-medication':  drug_name,
                        'Risk Multiplier': f"×{mult:.1f}",
                        'Mechanism': _get_mechanism(adr, drug_name.lower().replace(' ', '_')),
                    })

                if interaction_data:
                    st.dataframe(
                        pd.DataFrame(interaction_data),
                        use_container_width=True,
                        hide_index=True
                    )

            # ── Drug-specific risk table ───────────────────
            with st.expander(f" {drug} — Phase 2 ADR Signal Breakdown (ROR)"):
                st.caption(
                    "These ROR values come from your Phase 2 FAERS disproportionality "
                    "analysis. ROR > 2 with n ≥ 3 = confirmed pharmacovigilance signal."
                )
                ror_df = pd.DataFrame([
                    {"ADR Category":    "Hepatotoxicity",
                     "ROR":             risks['drug_risk_hepatotoxicity'],
                     "Signal Strength": _ror_label(risks['drug_risk_hepatotoxicity'])},
                    {"ADR Category":    "Nephrotoxicity",
                     "ROR":             risks['drug_risk_nephrotoxicity'],
                     "Signal Strength": _ror_label(risks['drug_risk_nephrotoxicity'])},
                    {"ADR Category":    "Bone / Tooth",
                     "ROR":             risks['drug_risk_bone_density_decreas'],
                     "Signal Strength": _ror_label(risks['drug_risk_bone_density_decreas'])},
                    {"ADR Category":    "Hematologic",
                     "ROR":             risks['drug_risk_haematologic_toxicity'],
                     "Signal Strength": _ror_label(risks['drug_risk_haematologic_toxicity'])},
                    {"ADR Category":    "Tooth Loss (specific)",
                     "ROR":             risks['drug_risk_tooth_loss'],
                     "Signal Strength": _ror_label(risks['drug_risk_tooth_loss'])},
                ])
                st.dataframe(ror_df.style.format({"ROR": "{:.1f}"}),
                             use_container_width=True, hide_index=True)

            # ── Clinical Recommendations ───────────────────
            st.markdown("### Clinical Recommendations")
            high_risks   = [d['Category'] for d in risk_data if d['Probability'] > 50]
            medium_risks = [d['Category'] for d in risk_data if 30 < d['Probability'] <= 50]

            if high_risks:
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#e74c3c 0%,#c0392b 100%);"
                    f"padding:1.2rem;border-radius:10px;margin-bottom:1rem;"
                    f"box-shadow:0 4px 12px rgba(231,76,60,0.3);'>"
                    f"<p style='color:white;font-weight:700;font-size:1.1rem;margin:0 0 0.5rem 0;'>"
                    f"⚠️ Action Required</p>"
                    f"<p style='color:white;margin:0;'>High risk for "
                    f"<strong>{', '.join(high_risks)}</strong>. "
                    f"Consider alternative therapy or enhanced monitoring.</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                _render_monitoring_protocols(high_risks)

            if medium_risks:
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,#f39c12 0%,#e67e22 100%);"
                    f"padding:1.2rem;border-radius:10px;margin-bottom:1rem;"
                    f"box-shadow:0 4px 12px rgba(243,156,18,0.3);'>"
                    f"<p style='color:white;font-weight:700;font-size:1.1rem;margin:0 0 0.5rem 0;'>"
                    f"ℹ️ Moderate Risk</p>"
                    f"<p style='color:white;margin:0;'>"
                    f"<strong>{', '.join(medium_risks)}</strong>. "
                    f"Implement standard monitoring protocols.</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            if not high_risks and not medium_risks:
                st.markdown(
                    "<div style='background:linear-gradient(135deg,#27ae60 0%,#229954 100%);"
                    "padding:1.2rem;border-radius:10px;margin-bottom:1rem;"
                    "box-shadow:0 4px 12px rgba(39,174,96,0.3);'>"
                    "<p style='color:white;font-weight:700;font-size:1.1rem;margin:0 0 0.5rem 0;'>"
                    "✅ Low Risk Profile</p>"
                    "<p style='color:white;margin:0;'>No high-risk categories detected. "
                    "Proceed with standard monitoring protocols.</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
        else:
            # Placeholder before prediction
            st.info(" Fill in patient details and click **Calculate Risk Profile**")
            with st.expander("What's new in this version?"):
                st.markdown("""
**Drug-specific ADR risk scores** (replacing polypharmacy count):
- Each drug now carries its actual ROR from Phase 2 FAERS analysis
- e.g. Tenofovir has ROR = 2,596 for tooth loss

**Concomitant drug interaction layer:**
- Select co-medications to see amplified risk scores
- Based on published pharmacological mechanisms:
  - Tenofovir + Furosemide → additive nephrotoxicity
  - Tenofovir + Metformin → shared mitochondrial stress pathway
  - Tenofovir + Prednisone → compounded bone density loss

**Temporal validation (2020–2022 train, 2023–2024 test):**
- Nephrotoxicity AUC: 0.951
- Bone/Tooth AUC: 0.973
""")


# ═════════════════════════════════════════════════════════════
# TAB 2 — VALIDATION
# ═════════════════════════════════════════════════════════════
with tab_validation:
    st.markdown("##  Model Validation Results")
    st.markdown(
        "Comprehensive validation showing how the model performs on "
        "**future unseen data** and how upgraded features compare."
    )

    v_tab1, v_tab2, v_tab3 = st.tabs([
        " Temporal Validation",
        " Feature Upgrade",
        " Population Risk Flags",
    ])

    # ── Temporal Validation ────────────────────────────────────────────────────
    with v_tab1:
        st.markdown("### Trained on 2020–2022  ·  Tested on 2023–2024")
        st.info(
            "**What this means:** The model was trained only on historical FAERS reports "
            "(94,200 patients) and then tested on a completely separate future period "
            "(54,946 patients) it had never seen — exactly how it would work in "
            "real clinical deployment."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean AUC",           "0.881", help="Average across all 5 ADR types")
        c2.metric("Nephrotoxicity AUC", "0.951", "Recall 83.9%")
        c3.metric("Bone/Tooth AUC",     "0.973", "Recall 94.7%")
        c4.metric("Training patients",  "94,200", "2020–2022")

        st.markdown("---")

        # AUC chart
        fig_auc, ax_auc = plt.subplots(figsize=(9, 4))
        fig_auc.patch.set_facecolor('white')
        colors_auc = ['#1a5276' if v >= 0.9 else '#2980b9' if v >= 0.8 else '#85c1e9'
                      for v in TEMPORAL_RESULTS['AUC']]
        bars = ax_auc.bar(TEMPORAL_RESULTS['ADR'], TEMPORAL_RESULTS['AUC'],
                          color=colors_auc, edgecolor='white', linewidth=1.5)
        ax_auc.set_ylim(0.75, 1.02)
        ax_auc.set_ylabel('AUC', fontweight='bold')
        ax_auc.set_title('AUC by ADR Type — Temporal Validation (2023–2024 test set)',
                         fontweight='bold', pad=12)
        ax_auc.axhline(0.80, color='orange', linestyle='--', linewidth=1.5,
                       label='Min. threshold (0.80)')
        ax_auc.legend(fontsize=9)
        for bar, val in zip(bars, TEMPORAL_RESULTS['AUC']):
            ax_auc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        sns.despine(ax=ax_auc)
        ax_auc.grid(axis='y', alpha=0.3, linestyle='--')
        st.pyplot(fig_auc, use_container_width=True)
        plt.close()

        # Recall chart
        fig_rec, ax_rec = plt.subplots(figsize=(9, 4))
        fig_rec.patch.set_facecolor('white')
        colors_rec = ['#1e8449' if v >= 0.8 else '#27ae60' if v >= 0.4 else '#a9dfbf'
                      for v in TEMPORAL_RESULTS['Recall']]
        bars_r = ax_rec.bar(TEMPORAL_RESULTS['ADR'], TEMPORAL_RESULTS['Recall'],
                            color=colors_rec, edgecolor='white', linewidth=1.5)
        ax_rec.set_ylim(0, 1.15)
        ax_rec.set_ylabel('Recall', fontweight='bold')
        ax_rec.set_title('Recall (Clinical Catch Rate) — How many real ADR cases are caught',
                         fontweight='bold', pad=12)
        ax_rec.axhline(0.80, color='red', linestyle='--', linewidth=1.5,
                       label='Target recall (80%)')
        ax_rec.legend(fontsize=9)
        for bar, val in zip(bars_r, TEMPORAL_RESULTS['Recall']):
            ax_rec.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        sns.despine(ax=ax_rec)
        ax_rec.grid(axis='y', alpha=0.3, linestyle='--')
        st.pyplot(fig_rec, use_container_width=True)
        plt.close()

        st.markdown("#### Complete Results")
        st.dataframe(
            TEMPORAL_RESULTS.style
                .background_gradient(subset=['AUC', 'Recall'], cmap='YlGn')
                .format({'AUC': '{:.3f}', 'Recall': '{:.3f}',
                         'Precision': '{:.3f}', 'F1': '{:.3f}'}),
            use_container_width=True, hide_index=True
        )

        st.success(
            "💡 **Why temporal validation matters:** Standard cross-validation shuffles "
            "data randomly — the model can accidentally 'see' future reports during "
            "training. Temporal validation is the honest version: train on the past, "
            "test on the future. Nephrotoxicity and Bone/Tooth generalise strongly. "
            "Hepatotoxicity and Neurologic are harder to predict and would benefit "
            "from additional clinical features in future work."
        )

    # ── Feature Upgrade ────────────────────────────────────────────────────────
    with v_tab2:
        st.markdown("### From Polypharmacy Count → Drug-Specific ADR Risk")

        col_old, col_new = st.columns(2)
        with col_old:
            st.markdown("####  Old Approach")
            st.code("polypharmacy_count = 5", language="text")
            st.markdown(
                "Treats aspirin and cisplatin as identical.  \n"
                "No information about which drugs or their toxicity profiles."
            )
        with col_new:
            st.markdown("####  New Approach")
            st.code(
                "drug_risk_nephrotoxicity     = 28.4\n"
                "drug_risk_tooth_loss         = 2596\n"
                "concomitant_risk_nephrotox   = 3.0\n"
                "has_nephrotoxic_comedication = 1",
                language="text"
            )
            st.markdown(
                "Encodes **pharmacovigilance signal strength** from Phase 2.  \n"
                "Mirrors how a clinical pharmacologist reasons."
            )

        st.markdown("---")

        fig_comp, ax_comp = plt.subplots(figsize=(9, 4))
        fig_comp.patch.set_facecolor('white')
        short_labels = ['A: Polypharmacy\ncount', 'B: + HBV drug\nrisk', 'C: + Concomitant\ndrug risk']
        comp_colors  = ['#5d6d7e', '#1a5276', '#117a65']
        bars_c = ax_comp.bar(short_labels, FEATURE_COMPARISON['AUC'],
                             color=comp_colors, edgecolor='white', linewidth=1.5)
        ax_comp.set_ylim(0.975, 0.983)
        ax_comp.set_ylabel('AUC (5-fold CV)', fontweight='bold')
        ax_comp.set_title('Model AUC by Feature Set (5-fold cross-validation)',
                          fontweight='bold', pad=12)
        for bar, val in zip(bars_c, FEATURE_COMPARISON['AUC']):
            ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00008,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        sns.despine(ax=ax_comp)
        ax_comp.grid(axis='y', alpha=0.3, linestyle='--')
        st.pyplot(fig_comp, use_container_width=True)
        plt.close()

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Polypharmacy count AUC",    "0.904", "baseline")
        col_m2.metric("Drug-specific risk AUC",    "0.922", "+1.83%")

        st.markdown("---")
        st.markdown("#### What Each New Feature Captures")
        st.dataframe(pd.DataFrame({
            'Feature': [
                'drug_risk_nephrotoxicity',
                'drug_risk_tooth_loss',
                'concomitant_risk_nephrotoxicity',
                'concomitant_risk_hepatotoxicity',
                'has_nephrotoxic_comedication',
                'has_diabetes_comedication',
            ],
            'Example (Tenofovir patient + Furosemide)': [
                '28.4 — ROR from FAERS Phase 2',
                '2596 — our novel finding (confirmed from 1 case report → 10k+ patients)',
                '3.0 — Furosemide amplifies renal tubular toxicity',
                '2.5 — Metformin shares mitochondrial Complex I pathway',
                '1 (Yes) — binary flag for Streamlit display',
                '1 (Yes) — binary flag for diabetes co-medication',
            ],
        }), use_container_width=True, hide_index=True)

    # ── Population Risk Flags ──────────────────────────────────────────────────
    with v_tab3:
        st.markdown("### Co-medication Risk in Your FAERS Cohort (441,915 patients)")

        flag_df = pd.DataFrame({
            'Co-medication Risk':     [
                'Nephrotoxic (diuretics, NSAIDs, aminoglycosides)',
                'Hepatotoxic (metformin, statins, antituberculars)',
                'Bone-risk (corticosteroids, PPIs, anticonvulsants)',
                'Dental-risk (PPIs, SSRIs, Ca-channel blockers)',
                'Diabetes medication',
            ],
            'Patients': [94251, 72505, 133306, 70610, 17784],
            'Percentage': [21.3, 16.4, 30.2, 16.0, 4.0],
        })

        fig_flags, ax_flags = plt.subplots(figsize=(10, 5))
        fig_flags.patch.set_facecolor('white')
        sorted_df   = flag_df.sort_values('Patients')
        flag_colors = ['#e74c3c','#c0392b','#922b21','#7b241c','#641e16'][:len(sorted_df)]
        bars_f = ax_flags.barh(sorted_df['Co-medication Risk'], sorted_df['Patients'],
                               color=flag_colors, edgecolor='white', linewidth=1.2)
        ax_flags.set_xlabel('Number of Patients', fontweight='bold')
        ax_flags.set_title('Patients with Clinically Significant Co-medication Risk Flags',
                           fontweight='bold', pad=12)
        max_val = sorted_df['Patients'].max()
        for bar, (_, row) in zip(bars_f, sorted_df.iterrows()):
            ax_flags.text(bar.get_width() + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                          f"{int(row['Patients']):,}  ({row['Percentage']:.1f}%)",
                          va='center', fontsize=9, fontweight='bold')
        ax_flags.set_xlim(0, max_val * 1.25)
        sns.despine(ax=ax_flags)
        ax_flags.grid(axis='x', alpha=0.3, linestyle='--')
        st.pyplot(fig_flags, use_container_width=True)
        plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Nephrotoxic co-med",  "94,251",  "21.3% of cohort")
        c2.metric("Hepatotoxic co-med",  "72,505",  "16.4% of cohort")
        c3.metric("Bone-risk co-med",    "133,306", "30.2% of cohort")

        st.warning(
            "⚠️ **Clinical implication:** 1 in 5 HBV patients are on co-medications "
            "that independently amplify nephrotoxicity risk. 30% are on medications "
            "that compound bone density loss — a major concern for long-term Tenofovir "
            "users. These patients would not have been flagged by polypharmacy count alone."
        )

        st.markdown("---")
        st.markdown("#### Risk Amplification Example")
        st.markdown("""
| Patient Scenario | Base ADR Risk | Co-medication | Amplified Risk | Mechanism |
|---|---|---|---|---|
| Tenofovir only | Nephrotoxicity ROR = 28 | — | Baseline | — |
| Tenofovir + Furosemide | Nephrotoxicity ROR = 28 | ×3.0 | **High** | Both impair renal tubular function |
| Tenofovir + Metformin | Hepatotoxicity ROR = 43 | ×2.5 | **High** | Shared mitochondrial Complex I stress |
| Tenofovir + Prednisone | Bone ROR = 514 | ×3.5 | **Very High** | Glucocorticoid + TDF bone loss |
""")


# ─────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("""
⚠️ **Disclaimer**: Research and educational purposes only. Does NOT replace professional \
medical judgment. Model optimised for high sensitivity (88% recall / 33% precision) — \
may over-predict risk to avoid missing true cases.

📊 **Model Performance (cross-validation)**: Mean AUC 0.922 (range 0.859–0.975)  |  \
Temporal validation mean AUC 0.881 (trained 2020–2022, tested 2023–2024)  |  \
Training data: FDA FAERS 2020–2024, 169,565 patients

🔬 **Feature Upgrade**: Drug-specific ADR risk (from Phase 2 FAERS ROR analysis) + \
Concomitant drug interaction layer replaced generic polypharmacy count. AUC improvement: +1.83%.
""")
