import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page title and layout
st.set_page_config(page_title="RenalCare AI Engine", layout="wide", page_icon="🫁")

# Custom CSS for the new Dark/Modern UI
st.markdown("""
<style>
    /* Theme Colors */
    :root {
        --primary: #9b51e0; /* Purple accent from image */
        --dark-bg: #121212;
        --text: #ffffff;
    }
    
    .main {
        background-color: #fcfcfc;
    }
    
    /* Hero Landing Section */
    .hero-container {
        background-color: var(--dark-bg);
        background-image: linear-gradient(to right, rgba(18,18,18,0.95), rgba(18,18,18,0.7)), url('https://img.freepik.com/free-vector/human-internal-organ-with-kidney_1308-111059.jpg');
        background-size: cover;
        background-position: center;
        padding: 8rem 5rem;
        border-radius: 0 0 100px 0;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::after {
        content: "";
        position: absolute;
        bottom: 0;
        right: 0;
        width: 300px;
        height: 300px;
        background: rgba(155, 81, 224, 0.2);
        filter: blur(100px);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        max-width: 500px;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Branding */
    .brand-header {
        position: absolute;
        top: 2rem;
        left: 5rem;
        font-size: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Custom Launch Button */
    .stButton>button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 0.8rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 20px rgba(155, 81, 224, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px rgba(155, 81, 224, 0.4) !important;
    }
    
    /* Bottom Section */
    .bottom-section {
        padding: 5rem 5rem;
        text-align: center;
    }
    
    .bottom-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #121212;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State for App Landing
if 'app_started' not in st.session_state:
    st.session_state.app_started = False

def start_app():
    st.session_state.app_started = True

# --- LANDING PAGE ---
if not st.session_state.app_started:
    # Modern Dark Landing UI inspired by the provided image
    st.markdown("""
        <div class="hero-container">
            <div class="brand-header">
                <span style="color:var(--primary); font-size: 2rem;">⚛️</span> RENALCARE
            </div>
            <h1 class="hero-title">Get Medical<br>Care early</h1>
            <p class="hero-subtitle">
                Advanced AI-driven diagnostics for Chronic Kidney Disease. 
                Early detection saves lives. Analyze your renal health using 
                precision machine learning models today.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Bottom Section with Launch Button
    st.markdown("""
        <div class="bottom-section">
            <h2 class="bottom-title">How to Protect Yourself</h2>
            <p style="color: #666; max-width: 700px; margin: 0 auto 3rem auto;">
                Regular checkups, monitoring Serum Creatinine, and calculating eGFR are vital steps 
                in maintaining kidney health. Our portal provides clinical-grade predictions 
                based on your unique lab values.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("LAUNCH DIAGNOSTIC APP"):
            start_app()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.stop()

# --- MAIN APP LOGIC ---
# Load models and preprocessing components
@st.cache_resource
def load_assets():
    try:
        hgb_model = joblib.load('models/ckd_hgb_model.joblib')
        rf_model = joblib.load('models/ckd_rf_model.joblib')
        imputer = joblib.load('models/imputer.joblib')
        features = joblib.load('models/features.joblib')
        return hgb_model, rf_model, imputer, features
    except FileNotFoundError:
        st.error("Error: Models and assets not found. Please run the training script first.")
        return None, None, None, None

hgb_model, rf_model, imputer, features = load_assets()

# Sidebar navigation
st.sidebar.title("CKD Predictor Portal")
app_mode = st.sidebar.selectbox("Choose the navigation mode:", ["Project Overview", "Manual Patient Prediction", "Bulk CSV Prediction"])

# Pre-defined patient diagnosis logic (based on the Colab rules)
def disease_rules(patient_dict):
    rules = [
        {"name": "End-Stage Renal Disease (ESRD)", "risk": 0, "reason": "Severe loss of kidney function (eGFR < 30)"},
        {"name": "Anemia of CKD", "risk": 0, "reason": "Low Hemoglobin levels associated with kidney stage"},
        {"name": "Heart Attack Risk (Myocardial Infarction)", "risk": 0, "reason": "History of Hypertension or CAD combined with high Blood Glucose"},
        {"name": "Congestive Heart Failure", "risk": 0, "reason": "Presence of Pedal Edema (swelling) and high Blood Pressure"},
        {"name": "Metabolic Disorder", "risk": 0, "reason": "High Blood Glucose (BDS) or Serum Creatinine levels"},
        {"name": "Urological Issues", "risk": 0, "reason": "Presence of Red Blood Cells or Bacteria in urine"}
    ]
    
    egfr = patient_dict.get('eGFR', 0)
    sc = patient_dict.get('sc', 0)
    hemo = patient_dict.get('hemo', 0)
    htn = patient_dict.get('htn', 0)
    cad = patient_dict.get('cad', 0)
    bgr = patient_dict.get('bgr', 0)
    rbc = patient_dict.get('rbc', 0)
    ba = patient_dict.get('ba', 0)
    pe = patient_dict.get('pe', 0)
    bp = patient_dict.get('bp', 80)
    
    # ── RENAL LOGIC ──
    if egfr < 15: rules[0]['risk'] = 92
    elif egfr < 30: rules[0]['risk'] = 65
    
    # ── HEMATOLOGY LOGIC (Anemia) ──
    if hemo < 10: rules[1]['risk'] = 85
    elif hemo < 12: rules[1]['risk'] = 45
    
    # ── HEART ATTACK RISK LOGIC ──
    h_risk = 0
    if htn == 1: h_risk += 30
    if cad == 1: h_risk += 40
    if bgr > 180: h_risk += 20
    if sc > 2.5: h_risk += 10
    rules[2]['risk'] = min(h_risk, 95)
    
    # ── HEART FAILURE LOGIC ──
    hf_risk = 0
    if pe == 1: hf_risk += 50
    if bp > 140: hf_risk += 20
    if egfr < 45: hf_risk += 15
    rules[3]['risk'] = min(hf_risk, 90)
    
    # ── METABOLIC LOGIC ──
    if bgr > 150: rules[4]['risk'] = 75
    elif sc > 2.0: rules[4]['risk'] = 60
    
    # ── UROLOGICAL LOGIC ──
    if rbc == 0 or ba == 1: rules[5]['risk'] = 80
    
    # Sort by risk
    return sorted([r for r in rules if r['risk'] > 0], key=lambda x: x['risk'], reverse=True)

if app_mode == "Project Overview":
    st.title("🫁 Chronic Kidney Disease (CKD) Prediction Dashboard")
    
    # Range of Stages Visualization
    st.markdown("### 📊 CKD Staging Guide (KDIGO 2022)")
    cols = st.columns(6)
    stages = [
        {"name": "G1", "range": "≥ 90", "desc": "Normal", "color": "#28a745"},
        {"name": "G2", "range": "60–89", "desc": "Mildly ↓", "color": "#94d066"},
        {"name": "G3a", "range": "45–59", "desc": "Mild-Mod ↓", "color": "#ffc107"},
        {"name": "G3b", "range": "30–44", "desc": "Mod-Sev ↓", "color": "#fd7e14"},
        {"name": "G4", "range": "15–29", "desc": "Severely ↓", "color": "#dc3545"},
        {"name": "G5", "range": "< 15", "desc": "Failure", "color": "#721c24"},
    ]
    for i, s in enumerate(stages):
        with cols[i]:
            st.markdown(f"""
                <div style="background-color:{s['color']}; padding:10px; border-radius:10px; text-align:center; color:white;">
                    <h4 style="margin:0;">{s['name']}</h4>
                    <p style="margin:0; font-size:0.8em;">{s['range']}</p>
                    <p style="margin:0; font-weight:bold;">{s['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    ### About the Project
    This system uses machine learning (HistGradientBoosting and Random Forest ensembles) to predict 
    the probability of a patient having Chronic Kidney Disease based on 25 biological and clinical parameters.
    
    #### Key Highlights:
    *   **Dual-Model Ensemble:** Combine HistGradientBoosting and Random Forest for high accuracy.
    *   **eGFR Integration:** Automated calculation using Serum Creatinine and Age for real-time staging.
    *   **Multi-Condition Analysis:** Predictions not just for CKD, but for future risks like Anemia, Cardiac issues, and Metabolic disorders.
    *   **Clinical Guidelines:** Logic based on KDIGO 2022 and ADA 2024.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/Stages_of_Kidney_Disease.jpg", width=600)

elif app_mode == "Manual Patient Prediction":
    st.title("🏥 Manual Patient Diagnosis Tool")
    st.write("Enter the patient's diagnostic values below to predict the risk of CKD and future complications.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (Years)", 1, 120, 50)
        bp = st.number_input("Blood Pressure (mm/Hg)", 50, 200, 80)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
        al = st.slider("Albumin (0-5)", 0, 5, 0)
        su = st.slider("Sugar (0-5)", 0, 5, 0)
    
    with col2:
        bgr = st.number_input("Blood Glucose Random (mgs/dl)", 50, 500, 120)
        bu = st.number_input("Blood Urea (mgs/dl)", 5, 400, 40)
        sc = st.number_input("Serum Creatinine (mgs/dl)", 0.1, 15.0, 1.2)
        sod = st.number_input("Sodium (mEq/L)", 100, 170, 138)
        pot = st.number_input("Potassium (mEq/L)", 2.0, 8.0, 4.2)
    
    with col3:
        hemo = st.number_input("Hemoglobin (gms)", 3.0, 20.0, 15.0)
        pcv = st.number_input("Packed Cell Volume (%)", 10, 60, 40)
        wc = st.number_input("White Blood Cell Count (cells/cmm)", 2000, 30000, 8000)
        rc = st.number_input("Red Blood Cell Count (millions/cmm)", 2.0, 8.0, 5.0)

    st.subheader("Clinical History & Labs")
    ca1, ca2, ca3, ca4 = st.columns(4)
    with ca1:
        htn = st.selectbox("Hypertension", ["no", "yes"])
        dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
    with ca2:
        cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
        pe = st.selectbox("Pedal Edema", ["no", "yes"])
    with ca3:
        ane = st.selectbox("Anemia", ["no", "yes"])
        appet = st.selectbox("Appetite", ["good", "poor"])
    with ca4:
        rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
        ba = st.selectbox("Bacteria", ["notpresent", "present"])

    if st.button("Analyze Patient Profile"):
        # Map categorical inputs
        patient_dict = {
            'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su, 'bgr': bgr, 'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot,
            'hemo': hemo, 'pcv': pcv, 'wc': wc, 'rc': rc,
            'htn': 1 if htn == 'yes' else 0, 'dm': 1 if dm == 'yes' else 0, 'cad': 1 if cad == 'yes' else 0,
            'pe': 1 if pe == 'yes' else 0, 'ane': 1 if ane == 'yes' else 0, 'appet': 1 if appet == 'good' else 0,
            'rbc': 1 if rbc == 'normal' else 0, 'pc': 1 if pc == 'normal' else 0,
            'pcc': 1 if pcc == 'present' else 0, 'ba': 1 if ba == 'present' else 0
        }
        
        # Calculate eGFR
        egfr = 175 * (sc ** -1.154) * (age ** -0.203)
        patient_dict['eGFR'] = egfr
        
        # Build features for prediction
        input_data = [patient_dict[f] for f in features] # type: ignore
        input_array = np.array(input_data).reshape(1, -1)
        input_imputed = imputer.transform(input_array) # type: ignore
        
        prob_hgb = hgb_model.predict_proba(input_imputed)[0][1] # pyright: ignore[reportOptionalMemberAccess]
        prob_rf = rf_model.predict_proba(input_imputed)[0][1] # type: ignore
        final_prob = (prob_hgb + prob_rf) / 2
        
        # Visualize Result
        st.divider()
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.metric("CKD Risk Probability", f"{final_prob*100:.1f}%")
            if final_prob > 0.5:
                st.error("Status: Chronic Kidney Disease Detected")
            else:
                st.success("Status: Low risk of CKD")
        
        with col_res2:
            st.metric("Estimated eGFR", f"{egfr:.1f} mL/min/1.73m²")
            if egfr >= 90: st.info("Stage: G1 - Normal")
            elif egfr >= 60: st.warning("Stage: G2 - Mildly Decreased")
            elif egfr >= 30: st.warning("Stage: G3 - Moderately Decreased")
            else: st.error("Stage: G4/G5 - Severely Decreased / Failure")

        st.subheader("Future Disease Projections")
        potential_diseases = disease_rules(patient_dict)
        
        # Display as horizontal widgets/cards
        if potential_diseases:
            # Create a row of cards for the top risks
            cols = st.columns(len(potential_diseases[:4])) # Show up to top 4 in columns
            for i, d in enumerate(potential_diseases[:4]):
                with cols[i]:
                    color = "#ff4b4b" if d['risk'] > 70 else "#fd7e14" if d['risk'] > 40 else "#28a745"
                    st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin-bottom: 10px; height: 100%;">
                            <h4 style="margin: 0; font-size: 0.9em; color: #333;">{d['name']}</h4>
                            <h2 style="margin: 5px 0; color: {color};">{d['risk']}%</h2>
                            <p style="margin: 0; font-size: 0.7em; color: #666;">{d['reason']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Additional details if any
            if len(potential_diseases) > 4:
                with st.expander("Show more risks"):
                    for d in potential_diseases[4:]:
                        st.write(f"- **{d['name']}**: {d['risk']}% (Reason: {d['reason']})")
        else:
            st.write("No significant future disease risks detected based on current profile.")

elif app_mode == "Bulk CSV Prediction":
    st.title("📂 Automated Batch Prediction")
    st.write("Upload a patient dataset in CSV format (containing at least the basic features like age, bp, sc, hemo, etc.).")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(data.head())
        
        if st.button("Run Batch Prediction"):
            # Progress bar for better UX
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Preprocessing
            status_text.text("🔄 Cleaning data and calculating eGFR...")
            temp_df = data.copy()
            
            # Apply cleaning (same as train.py)
            temp_df = temp_df.replace(to_replace={'\t': '', '\?': np.nan}, regex=True)
            for col in temp_df.columns:
                if temp_df[col].dtype == 'object' and hasattr(temp_df[col], 'str'):
                    temp_df[col] = temp_df[col].str.strip()

            # Mapping categorical
            mapper = {'yes': 1, 'no': 0, 'good': 1, 'poor': 0, 'normal': 1, 'abnormal': 0, 'present': 1, 'notpresent': 0}
            cat_cols = ['htn', 'dm', 'cad', 'pe', 'ane', 'appet', 'rbc', 'pc', 'pcc', 'ba']
            for col in cat_cols:
                if col in temp_df.columns:
                    temp_df[col] = temp_df[col].map(mapper)
            
            # Target map
            if 'classification' in temp_df.columns:
                temp_df['classification'] = temp_df['classification'].map({'ckd': 1, 'notckd': 0, 'ckd\t': 1})

            # Calculate eGFR for each row
            if 'sc' in temp_df.columns and 'age' in temp_df.columns:
                temp_df['eGFR'] = 175 * (temp_df['sc'].astype(float) ** -1.154) * (temp_df['age'].astype(float) ** -0.203)
            
            # Predict
            status_text.text("🤖 Running Ensemble Model...")
            X_input = temp_df[features]
            X_imputed = imputer.transform(X_input) # type: ignore
            
            p_hgb = hgb_model.predict_proba(X_imputed)[:, 1] # pyright: ignore[reportOptionalMemberAccess]
            p_rf = rf_model.predict_proba(X_imputed)[:, 1] # type: ignore
            final_probs = (p_hgb + p_rf) / 2
            
            temp_df['CKD_Probability'] = [f"{p*100:.1f}%" for p in final_probs]
            temp_df['CKD_Status'] = ["CKD Detected" if p > 0.5 else "Low Risk" for p in final_probs]
            
            # Future Disease Logic
            status_text.text("🏥 Analyzing future disease risks...")
            all_complications = []
            for idx, row in temp_df.iterrows():
                p_dict = row.to_dict()
                complications = disease_rules(p_dict)
                comp_str = "; ".join([f"{c['name']} ({c['risk']}%)" for c in complications])
                all_complications.append(comp_str if comp_str else "None")
                progress_bar.progress((idx + 1) / len(temp_df)) # type: ignore
            
            temp_df['Future_Complications'] = all_complications
            
            status_text.success("✅ Analysis complete!")
            st.write("### Prediction Results")
            # Show key result columns first
            cols_to_show = ['CKD_Status', 'CKD_Probability', 'eGFR', 'Future_Complications'] + [c for c in temp_df.columns if c not in ['CKD_Status', 'CKD_Probability', 'eGFR', 'Future_Complications']]
            st.dataframe(temp_df[cols_to_show])
            
            st.download_button("📥 Download Detailed Report", temp_df.to_csv(index=False), "ckd_batch_predictions.csv")
            
            # Show summary widgets for the first 3 patients in the batch
            st.divider()
            st.markdown("### 🏥 Batch Risk Highlights (First 3 Patients)")
            for i, row in temp_df.head(3).iterrows():
                st.markdown(f"**Patient Record {i}**")
                p_dict = row.to_dict()
                complications = disease_rules(p_dict)
                if complications:
                    cols = st.columns(len(complications[:3]))
                    for j, d in enumerate(complications[:3]):
                        with cols[j]:
                            color = "#ff4b4b" if d['risk'] > 70 else "#fd7e14" if d['risk'] > 40 else "#28a745"
                            st.markdown(f"""
                                <div style="background-color: white; padding: 12px; border-radius: 8px; border-left: 5px solid {color}; box-shadow: 2px 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px;">
                                    <h5 style="margin: 0; font-size: 0.85em; color: #333;">{d['name']}</h5>
                                    <h3 style="margin: 5px 0; color: {color};">{d['risk']}%</h3>
                                    <p style="margin: 0; font-size: 0.65em; color: #666;">{d['reason']}</p>
                                </div>
                            """, unsafe_allow_html=True)
                else: 
                    st.write("No significant risks detected for this patient.")
                st.divider()
