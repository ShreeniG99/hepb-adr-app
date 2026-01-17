import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Page Configuration
st.set_page_config(
    page_title="HepB ADR Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load and encode background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load background image
bg_image_path = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis\bg_pic.png"

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
        
        /* Add overlay for better text readability */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.85);
            z-index: -1;
        }}
    </style>
    """
else:
    # Fallback to solid color if image not found
    bg_img_css = """
    <style>
        .stApp {
            background-color: #f0f4f8;
        }
    </style>
    """

st.markdown(bg_img_css, unsafe_allow_html=True)

# Custom CSS for Medical Theme (Navy Blue)
st.markdown("""
<style>
    :root {
        --primary-color: #002244; /* Navy Blue */
        --secondary-color: #004488; /* Lighter Navy */
        --accent-color: #e74c3c; /* Alert Red */
        --text-color: #2c3e50; /* Dark Slate */
    }
    
    .main-header {
        background: linear-gradient(135deg, #002244 0%, #004488 100%);
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #ecf0f1;
        font-size: 1.1rem;
    }
    
    .stButton>button {
        background-color: var(--secondary-color);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--primary-color);
        box-shadow: 0 4px 12px rgba(0,34,68,0.2);
        transform: translateY(-2px);
    }
    
    .result-card {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
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
    
    .risk-low { border-left-color: #2ecc71 !important; }
    .risk-medium { border-left-color: #f1c40f !important; }
    .risk-high { border-left-color: #e74c3c !important; }
    
    h2, h3 {
        color: var(--secondary-color);
        font-family: 'Segoe UI', sans-serif;
    }
    
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
    }
    
    /* Make input containers semi-transparent */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.9);
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load Model (Cached)
@st.cache_resource
def load_model_resources():
    # Try relative path first
    model_path = "phase3_models/adr_classifier.pkl"
    scaler_path = "phase3_models/feature_scaler.pkl"
    
    # If not found, try absolute path
    if not os.path.exists(model_path):
        base_dir = r"C:\Users\shree\OneDrive\2nd yr\ADR-hepatatis"
        model_path = os.path.join(base_dir, "phase3_models", "adr_classifier.pkl")
        scaler_path = os.path.join(base_dir, "phase3_models", "feature_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Feature Definitions
ADR_CATEGORIES = [
    'hepatotoxicity', 'nephrotoxicity', 'bone_tooth', 
    'hematologic', 'neurologic'
]

FEATURE_COLS = [
    # Demographics
    'age_years', 'sex_binary', 'sex_female',
    'reporter_md', 'reporter_lawyer', 'reporter_consumer',
    
    # Severity (Zeroed out for prediction)
    'outcome_death', 'outcome_life_threat', 'outcome_hospitalization',
    'outcome_disability', 'severe_disease', 'severity_score',
    
    # Polypharmacy
    'drug_count', 'polypharmacy_low', 'polypharmacy_medium', 'polypharmacy_high',
    
    # Cohort
    'cohort_hbv_mono', 'cohort_hiv_hbv', 'cohort_autoimmune',
    
    # Drug
    'drug_entecavir', 'drug_tenofovir', 'drug_lamivudine', 
    'drug_adefovir', 'drug_telbivudine', 'drug_interferon', 'drug_risk_score'
]

# Drug Risk Scores (Hardcoded from analysis)
DRUG_RISK_SCORES = {
    'ADEFOVIR': 24.5285,
    'ENTECAVIR': 8.5138,
    'INTERFERON': 11.2757, 
    'LAMIVUDINE': 6.2125,
    'TELBIVUDINE': 284.8836,
    'TENOFOVIR': 4.0220
}

# --- Main App Layout ---

# Header
st.markdown("""
<div class="main-header">
    <h1>Clinical ADR Predictor</h1>
    <p>Hepatitis B Adverse Drug Reaction Risk Assessment System</p>
</div>
""", unsafe_allow_html=True)

# Load resources
model, scaler = load_model_resources()

if model is None:
    st.error("🚨 Critical Error: Model files not found. Please ensure 'adr_classifier.pkl' and 'feature_scaler.pkl' are in the 'phase3_models' directory.")
    st.stop()

# Input Section (Sidebar + Main Column)
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Patient Data")
    
    with st.container(border=True):
        # Demographics
        age = st.slider("Patient Age", 18, 90, 45, help="Patient age in years")
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        
        # Clinical Context
        st.markdown("#### Clinical Context")
        cohort = st.selectbox(
            "Patient Cohort",
            ["HBV Monoinfection", "HIV/HBV Coinfection", "HBV + Autoimmune", "HBV Other"]
        )
        
        reporter = st.selectbox(
            "Reporter Type", 
            ["Healthcare Professional", "Consumer", "Lawyer"],
            index=0,
            help="Source of the adverse event report"
        )
        
        # Treatment
        st.markdown("#### Treatment Plan")
        drug = st.selectbox(
            "Antiviral Drug",
            ["TENOFOVIR", "ENTECAVIR", "LAMIVUDINE", "ADEFOVIR", "TELBIVUDINE", "INTERFERON"]
        )
        
        polypharmacy = st.select_slider(
            "Polypharmacy Level",
            options=["Low (1-2)", "Medium (3-5)", "High (6+)"],
            value="Low (1-2)",
            help="Number of concomitant medications"
        )

    # Predict Button
    predict_btn = st.button("🔬 Calculate Risk Profile", use_container_width=True)

# Processing Logic
if predict_btn:
    # 1. Initialize Feature Vector
    features = pd.DataFrame(0, index=[0], columns=FEATURE_COLS)
    
    # 2. Map Inputs to Features
    
    # Demographics
    features['age_years'] = age
    features['sex_binary'] = 1 if sex == "Male" else 0
    features['sex_female'] = 1 if sex == "Female" else 0
    
    # Reporter
    if "Healthcare" in reporter or "Professional" in reporter:
        features['reporter_md'] = 1
    elif "Lawyer" in reporter:
        features['reporter_lawyer'] = 1
    else:
        features['reporter_consumer'] = 1
    
    # Severity (ALL ZEROS - no outcomes yet, we're predicting!)
    features['severity_score'] = 1  # Baseline
    
    # Polypharmacy
    if "Low" in polypharmacy:
        features['drug_count'] = 1.5
        features['polypharmacy_low'] = 1
    elif "Medium" in polypharmacy:
        features['drug_count'] = 4.0
        features['polypharmacy_medium'] = 1
    else:
        features['drug_count'] = 7.0
        features['polypharmacy_high'] = 1
        
    # Cohort
    if "Monoinfection" in cohort:
        features['cohort_hbv_mono'] = 1
    elif "HIV" in cohort:
        features['cohort_hiv_hbv'] = 1
    elif "Autoimmune" in cohort:
        features['cohort_autoimmune'] = 1
    # HBV Other: all cohort features remain 0 (reference category)
        
    # Drug
    features[f'drug_{drug.lower()}'] = 1
    features['drug_risk_score'] = DRUG_RISK_SCORES.get(drug, 1.0)
    
    # 3. Scale Features
    features_scaled = scaler.transform(features)
    
    # 4. Predict
    probas = []
    try:
        raw_probas = model.predict_proba(features_scaled)
        for p in raw_probas:
            probas.append(p[0][1])
    except:
        st.error("Error generating probabilities.")
        st.stop()
        
    # 5. Visualize Results
    with col2:
        st.markdown("### Risk Assessment Results")
        
        # Patient summary card
        with st.container():
            st.markdown(f"""
            <div style="background-color: rgba(255, 255, 255, 0.95); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <h4 style="margin: 0; color: #002244;">Patient Profile</h4>
                <p style="margin: 0.5rem 0 0 0; color: #2c3e50;">
                    <strong>{age} year old {sex}</strong> | {cohort} | {drug} | {polypharmacy} drugs
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Grid for cards
        grid_col1, grid_col2 = st.columns(2)
        
        risk_data = []
        
        for idx, (category, prob) in enumerate(zip(ADR_CATEGORIES, probas)):
            category_name = category.replace('_', ' ').title()
            risk_percent = prob * 100
            
            # Determine Risk Level
            if risk_percent > 50:
                risk_level = "High"
                risk_class = "risk-high"
                icon = "🔴"
                color = "#e74c3c"
            elif risk_percent > 30:
                risk_level = "Medium"
                risk_class = "risk-medium"
                icon = "🟡"
                color = "#f1c40f"
            else:
                risk_level = "Low"
                risk_class = "risk-low"
                icon = "🟢"
                color = "#2ecc71"
            
            risk_data.append({
                "Category": category_name, 
                "Probability": risk_percent,
                "Color": color
            })
            
            # Display Card - Category name in BLACK
            with (grid_col1 if idx % 2 == 0 else grid_col2):
                st.markdown(f"""
                <div class="result-card {risk_class}">
                    <h3 style="margin:0; font-size:1rem; color: black; font-weight: 700; background: color;">{category_name}</h3>
                    <div style="display:flex; align-items:baseline; justify-content:space-between; margin-top:0.5rem;">
                        <span style="font-size:1.8rem; font-weight:bold; color:{color};">{risk_percent:.1f}%</span>
                        <span style="font-size:0.9rem; color: black; font-weight:600;">{icon} {risk_level} Risk</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Chart with WHITE axis labels
        st.markdown("#### Comparative Risk Analysis")
        df_risk = pd.DataFrame(risk_data)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0.0)  # Transparent figure background
        ax.patch.set_alpha(0.95)  # Semi-transparent plot background
        ax.patch.set_facecolor('white')
        
        sns.barplot(data=df_risk, x="Probability", y="Category", palette=[d['Color'] for d in risk_data], ax=ax)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontweight='bold', color='white')
        ax.set_ylabel("", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        sns.despine(left=True, bottom=True)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        st.pyplot(fig)
        
        # Recommendations - ALL WITH WHITE TEXT ON COLORED BACKGROUNDS
        st.markdown("### Clinical Recommendations")
        
        high_risks = [d['Category'] for d in risk_data if d['Probability'] > 50]
        medium_risks = [d['Category'] for d in risk_data if 30 < d['Probability'] <= 50]
        
        if high_risks:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                        padding: 1.2rem; border-radius: 10px; margin-bottom: 1rem;
                        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0 0 0.5rem 0;">
                    ⚠️ Action Required
                </p>
                <p style="color: white; margin: 0; font-size: 1rem;">
                    High risk detected for <strong>{', '.join(high_risks)}</strong>. 
                    Consider alternative therapies or enhanced monitoring.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.95); padding: 1rem; 
                        border-radius: 10px; border-left: 4px solid #e74c3c; margin-bottom: 1rem;">
                <h4 style="color: #002244; margin: 0 0 0.75rem 0; font-size: 1.1rem;">🔍 Specific Monitoring Protocols:</h4>
            """, unsafe_allow_html=True)
            
            # Specific recommendations with colored backgrounds
            if "Hepatotoxicity" in high_risks:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e67e22 0%, #d35400 100%); 
                            padding: 0.8rem; border-radius: 8px; margin-bottom: 0.75rem;">
                    <p style="color: white; margin: 0; font-weight: 600;">
                        <strong>Hepatotoxicity:</strong> Monitor liver enzymes (ALT, AST) monthly. 
                        Watch for signs of hepatic decompensation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            if "Nephrotoxicity" in high_risks:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                            padding: 0.8rem; border-radius: 8px; margin-bottom: 0.75rem;">
                    <p style="color: white; margin: 0; font-weight: 600;">
                        <strong>Nephrotoxicity:</strong> Monitor kidney function (eGFR, creatinine) every 3 months. 
                        Consider dose adjustment if eGFR < 50 mL/min.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            if "Bone Tooth" in high_risks:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); 
                            padding: 0.8rem; border-radius: 8px; margin-bottom: 0.75rem;">
                    <p style="color: white; margin: 0; font-weight: 600;">
                        <strong>Bone/Tooth:</strong> DEXA scan annually. Recommend Calcium/Vitamin D supplementation. 
                        Dental exam every 6 months.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            if "Hematologic" in high_risks:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                            padding: 0.8rem; border-radius: 8px; margin-bottom: 0.75rem;">
                    <p style="color: white; margin: 0; font-weight: 600;">
                        <strong>Hematologic:</strong> Complete blood count (CBC) every 2 months. 
                        Monitor for cytopenias.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            if "Neurologic" in high_risks:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #16a085 0%, #138d75 100%); 
                            padding: 0.8rem; border-radius: 8px; margin-bottom: 0.75rem;">
                    <p style="color: white; margin: 0; font-weight: 600;">
                        <strong>Neurologic:</strong> Assess for peripheral neuropathy symptoms. 
                        Consider neurological consultation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        if medium_risks:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); 
                        padding: 1.2rem; border-radius: 10px; margin-bottom: 1rem;
                        box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0 0 0.5rem 0;">
                    ℹ️ Moderate Risk
                </p>
                <p style="color: white; margin: 0; font-size: 1rem;">
                    <strong>{', '.join(medium_risks)}</strong>. Implement standard monitoring protocols.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        if not high_risks and not medium_risks:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #27ae60 0%, #229954 100%); 
                        padding: 1.2rem; border-radius: 10px; margin-bottom: 1rem;
                        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);">
                <p style="color: white; font-weight: 700; font-size: 1.1rem; margin: 0 0 0.5rem 0;">
                    ✅ Low Risk Profile
                </p>
                <p style="color: white; margin: 0; font-size: 1rem;">
                    No high-risk categories detected. Proceed with standard monitoring protocols.
                </p>
            </div>
            """, unsafe_allow_html=True)

# Disclaimer Footer
st.divider()
st.caption("""
⚠️ **Disclaimer**: This tool is an AI decision support system for **research and educational purposes only**. 
It does **NOT replace professional medical judgment**. The model is optimized for high sensitivity (88% recall) 
with moderate precision (33%), meaning it may over-predict risk to avoid missing true cases. 
Training data: FDA FAERS (2020-2024), 169,565 patients.

📊 **Model Performance**: Mean Test AUC 0.922 (Range: 0.859-0.975) across 5 ADR categories
- Hepatotoxicity: AUC 0.919
- Nephrotoxicity: AUC 0.952  
- Bone/Tooth: AUC 0.975
- Hematologic: AUC 0.905
- Neurologic: AUC 0.859
""")