import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

# Add the project directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import our modules
from utils.data_preprocessing import DataPreprocessor
from models.model_training import ReadmissionPredictor

# Set page configuration
st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    /* Main container styling */
    .main-container {
        background: white;
        min-height: 100vh;
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 500;
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.5rem;
    }
    
    /* Risk indicators */
    .risk-high {
        color: #d13838;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .risk-low {
        color: inherit;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Info and warning boxes */
    .info-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4472c4;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9900;
        margin: 1rem 0;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        background: white;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 1px solid #ccc;
    }
    
    /* Buttons styling */
    .stButton>button {
        background-color: transparent;
        color: inherit;
        border: 1px solid #ccc;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #f0f0f0;
    }
    

    
    /* Input fields styling */
    div[data-baseweb="select"] > div {
        background-color: #f9f9f9;
        border: 1px solid #ccc;
        border-radius: 0.25rem;
    }
    
    div[data-baseweb="input"] > div {
        background-color: #f9f9f9;
        border: 1px solid #ccc;
        border-radius: 0.25rem;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: inherit;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #ccc;
        border-radius: 0.25rem;
        overflow: hidden;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🏥 Hospital Readmission Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Advanced Machine Learning Model for Predicting 30-Day Hospital Readmission</p>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_predictor' not in st.session_state:
    st.session_state.trained_predictor = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'X_train_scaled' not in st.session_state:
    st.session_state.X_train_scaled = None

# Sidebar
with st.sidebar:
    st.header("ℹ️ About This Application")
    st.info("""
    This application predicts the risk of 30-day hospital readmission based on patient information.
    
    **How it works:**
    1. Enter patient details in the input fields
    2. Train the models (if not already done)
    3. Select the model to use for prediction
    4. Click 'Predict' to get the risk assessment
    """)
    
    st.header("📊 Navigate Pages")
    st.write("""
    - **Risk Prediction**: Patient-level risk assessment
    - **Analytics Dashboard**: Hospital-level analytics
    """)
    
    st.header("🧠 Model Information")
    st.write("**Algorithms Used:**")
    st.write("- Logistic Regression")
    st.write("- Random Forest")
    

    
    st.header("🩺 Diagnosis Code Reference")
    st.write("**250:** Diabetes mellitus")
    st.write("**428:** Heart failure")
    st.write("**414:** Coronary atherosclerosis")
    st.write("**403:** Hypertensive chronic kidney disease")
    st.write("**276:** Disorders of fluid, electrolyte, and acid-base balance")
    st.write("**427:** Cardiac dysrhythmias")
    st.write("**584:** Acute kidney failure")
    st.write("**250.01:** Diabetes with ketoacidosis")
    st.write("**?:** Unknown/Not specified")
    
    st.header("🏥 Admission Type Reference")
    st.write("**1:** Emergency")
    st.write("**2:** Urgent")
    st.write("**3:** Elective")
    st.write("**4:** Newborn")
    st.write("**5:** Trauma Center")
    st.write("**6:** Transfer from hospital")
    st.write("**7:** Transfer from skilled nursing facility")
    st.write("**8:** Transfer from intermediate care facility")
    
    st.header("🏥 Admission Source Reference")
    st.write("**1:** Physician referral")
    st.write("**2:** Clinic referral")
    st.write("**3:** HMO referral")
    st.write("**4:** Transfer from hospital")
    st.write("**5:** Transfer from skilled nursing facility")
    st.write("**6:** Transfer from another health care facility")
    st.write("**7:** Emergency room")
    st.write("**8:** Court/law enforcement")
    st.write("**9:** Information not available")
    st.write("**10:** Transfer from critical access hospital")
    st.write("**11:** Normal delivery")
    st.write("**12:** Premature delivery")
    st.write("**13:** Sick baby")
    st.write("**14:** Extramural birth")
    st.write("**15:** Not available")
    st.write("**16:** Transfer from trauma center")
    st.write("**17:** Transfer from rehabilitation fac or another health")
    st.write("**18:** Transfer from psychiatric hospital")
    
    st.header("🏥 Discharge Disposition Reference")
    st.write("**1:** Discharged to home")
    st.write("**2:** Discharged/transferred to another short term general hospital")
    st.write("**3:** Discharged/transferred to SNF (Skilled Nursing Facility)")
    st.write("**4:** Discharged/transferred to ICF (Intermediate Care Facility)")
    st.write("**5:** Discharged/transferred to another type of institution")
    st.write("**6:** Discharged to home with home health service")
    st.write("**7:** Left against medical advice")
    st.write("**8:** Expired (Deceased)")
    st.write("**9:** Discharged/transferred to home under care of Home IV provider")
    st.write("**10:** Discharged/transferred to a federal health care facility")
    st.write("**11:** Discharged/transferred to a psychiatric hospital")
    st.write("**12:** Discharged/transferred to a critical access hospital")
    st.write("**13:** Discharged to court/law enforcement")
    st.write("**14:** Discharged to probation/parole office")
    st.write("**15:** Discharged to rehabilitation facility")
    st.write("**16:** Discharged to nursing facility certified under Medicaid")
    st.write("**17:** Discharged to another type of health care institution")
    st.write("**18:** Discharged to home under care of organized home health service")
    st.write("**19:** Discharged/transferred to Medicare certified nursing facility")
    st.write("**20:** Hospice medical facility")
    st.write("**21:** Hospice home")
    st.write("**22:** Discharged/transferred to another rehab facility")
    st.write("**23:** Discharged/transferred to a psychiatric facility certified under Medicare/Medicaid")
    st.write("**24:** Discharged/transferred to Oxygency/IV service")
    st.write("**25:** Discharged to care of family under arrangement of a Home Health Agency")
    st.write("**26:** Discharged/transferred to another type of health care facility")
    st.write("**27:** Discharged/transferred to a nursing home certified under Medicaid")
    st.write("**28:** Discharged to home with public health nurse")
    st.write("**29:** Other")
    
    st.header("👥 Social Risk Factors Reference")
    st.write("**0:** No significant social risk factors")
    st.write("**1:** Minor social challenges (e.g., occasional transportation issues)")
    st.write("**2:** Moderate social challenges (e.g., housing instability, limited family support)")
    st.write("**3:** Significant social challenges (e.g., homelessness, substance abuse, severe mental health issues)")
    st.write("**4:** Severe social challenges with multiple compounding factors")
    st.info("Social risk factors significantly impact readmission rates by affecting patient's ability to adhere to treatment plans, attend follow-up appointments, and manage their health conditions effectively.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 class='sub-header'>📋 Patient Information</h2>", unsafe_allow_html=True)
    
    # Create input fields for patient data
    with st.form(key='patient_form'):
        # Demographics
        st.subheader("👤 Demographics")
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            gender = st.selectbox("Gender", ["Male", "Female", "Unknown/Invalid"])
            age_group = st.selectbox("Age Group", 
                                   ['[0-10)', '[10-20)', '[20-30)', '[30-40)', 
                                    '[40-50)', '[50-60)', '[60-70)', '[70-80)', 
                                    '[80-90)', '[90-100)'])
        
        with col_demo2:
            race = st.selectbox("Race", 
                              ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "?"])
        
        # Admission Information
        st.subheader("🏥 Admission Details")
        
        # Dictionary mapping admission type codes to descriptions
        admission_type_map = {
            1: 'Emergency',
            2: 'Urgent',
            3: 'Elective',
            4: 'Newborn',
            5: 'Trauma Center',
            6: 'Transfer from hospital',
            7: 'Transfer from skilled nursing facility',
            8: 'Transfer from intermediate care facility'
        }
        
        # Dictionary mapping admission source codes to descriptions
        admission_source_map = {
            1: 'Physician referral',
            2: 'Clinic referral',
            3: 'HMO referral',
            4: 'Transfer from hospital',
            5: 'Transfer from skilled nursing facility',
            6: 'Transfer from another health care facility',
            7: 'Emergency room',
            8: 'Court/law enforcement',
            9: 'Information not available',
            10: 'Transfer from critical access hospital',
            11: 'Normal delivery',
            12: 'Premature delivery',
            13: 'Sick baby',
            14: 'Extramural birth',
            15: 'Not available',
            16: 'Transfer from trauma center',
            17: 'Transfer from rehabilitation fac or another health',
            18: 'Transfer from psychiatric hospital'
        }
        
        # Dictionary mapping discharge disposition codes to descriptions
        discharge_disposition_map = {
            1: 'Discharged to home',
            2: 'Discharged/transferred to another short term general hospital',
            3: 'Discharged/transferred to SNF (Skilled Nursing Facility)',
            4: 'Discharged/transferred to ICF (Intermediate Care Facility)',
            5: 'Discharged/transferred to another type of institution',
            6: 'Discharged to home with home health service',
            7: 'Left against medical advice',
            8: 'Expired (Deceased)',
            9: 'Discharged/transferred to home under care of Home IV provider',
            10: 'Discharged/transferred to a federal health care facility',
            11: 'Discharged/transferred to a psychiatric hospital',
            12: 'Discharged/transferred to a critical access hospital',
            13: 'Discharged to court/law enforcement',
            14: 'Discharged to probation/parole office',
            15: 'Discharged to rehabilitation facility',
            16: 'Discharged to nursing facility certified under Medicaid',
            17: 'Discharged to another type of health care institution',
            18: 'Discharged to home under care of organized home health service',
            19: 'Discharged/transferred to Medicare certified nursing facility',
            20: 'Hospice medical facility',
            21: 'Hospice home',
            22: 'Discharged/transferred to another rehab facility',
            23: 'Discharged/transferred to a psychiatric facility certified under Medicare/Medicaid',
            24: 'Discharged/transferred to Oxygency/IV service',
            25: 'Discharged to care of family under arrangement of a Home Health Agency',
            26: 'Discharged/transferred to another type of health care facility',
            27: 'Discharged/transferred to a nursing home certified under Medicaid',
            28: 'Discharged to home with public health nurse',
            29: 'Other'
        }
        
        col_adm1, col_adm2 = st.columns(2)
        
        with col_adm1:
            admission_type_code = st.selectbox("Admission Type", list(admission_type_map.keys()))
            st.caption(f"*{admission_type_map[admission_type_code]}*")
            discharge_disposition_code = st.selectbox("Discharge Disposition", list(discharge_disposition_map.keys()))
            st.caption(f"*{discharge_disposition_map[discharge_disposition_code]}*")
        
        with col_adm2:
            admission_source_code = st.selectbox("Admission Source", list(admission_source_map.keys()))
            st.caption(f"*{admission_source_map[admission_source_code]}*")
            time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
            
        # Additional Admission Information
        st.subheader("🏥 Additional Admission Details")
        col_add1, col_add2 = st.columns(2)
        
        with col_add1:
            prior_admissions = st.slider("Number of Prior Admissions (Last 12 Months)", 0, 20, 0)
            
        with col_add2:
            insurance_type = st.selectbox("Insurance Type", 
                                        ["Private", "Medicaid", "Medicare", "Self-Pay", "Government", "Other"])
        
        # Medical Information
        st.subheader("⚕️ Medical Details")
        col_med1, col_med2 = st.columns(2)
        
        with col_med1:
            num_medications = st.slider("Number of Medications", 1, 25, 10)
            num_lab_procedures = st.slider("Number of Lab Procedures", 1, 50, 15)
            num_procedures = st.slider("Number of Procedures", 0, 5, 1)
        
        with col_med2:
            number_outpatient = st.slider("Outpatient Visits", 0, 10, 2)
            number_emergency = st.slider("Emergency Visits", 0, 10, 1)
            number_inpatient = st.slider("Inpatient Visits", 0, 10, 1)
            number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)
            
        # Comorbidities
        st.subheader("🩺 Comorbidities")
        col_comorb = st.columns(3)
        
        with col_comorb[0]:
            chf = st.checkbox("Congestive Heart Failure (CHF)")
            copd = st.checkbox("Chronic Obstructive Pulmonary Disease (COPD)")
            
        with col_comorb[1]:
            diabetes = st.checkbox("Diabetes")
            bp = st.checkbox("Hypertension (BP)")
            
        with col_comorb[2]:
            ckd = st.checkbox("Chronic Kidney Disease (CKD)")
            other_comorb = st.checkbox("Other")
        
        # Diagnosis Information
        st.subheader("🩺 Diagnosis Codes")
        
        # Dictionary mapping diagnosis codes to disease names
        diag_code_map = {
            '250': 'Diabetes mellitus',
            '428': 'Heart failure',
            '414': 'Coronary atherosclerosis',
            '403': 'Hypertensive chronic kidney disease',
            '276': 'Disorders of fluid, electrolyte, and acid-base balance',
            '427': 'Cardiac dysrhythmias',
            '584': 'Acute kidney failure',
            '250.01': 'Diabetes with ketoacidosis',
            '?': 'Unknown/Not specified'
        }
        
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        
        with col_diag1:
            diag_1_code = st.selectbox("Primary Diagnosis", 
                                      list(diag_code_map.keys()))
            st.caption(f"*{diag_code_map[diag_1_code]}*")
            
        with col_diag2:
            diag_2_code = st.selectbox("Secondary Diagnosis", 
                                      list(diag_code_map.keys()))
            st.caption(f"*{diag_code_map[diag_2_code]}*")
            
        with col_diag3:
            diag_3_code = st.selectbox("Additional Diagnosis", 
                                      list(diag_code_map.keys()))
            st.caption(f"*{diag_code_map[diag_3_code]}*")
        
        # Medications
        st.subheader("💊 Medication Status")
        col_med_status = st.columns(2)
        
        with col_med_status[0]:
            metformin = st.selectbox("Metformin", ["No", "Up", "Down", "Steady"])
        with col_med_status[1]:
            insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"])
        
        # Enhanced Clinical Indicators
        st.subheader("🩺 Enhanced Clinical Indicators")
        col_clinical1, col_clinical2 = st.columns(2)
        
        with col_clinical1:
            hba1c_result = st.selectbox("HbA1c Result", ["None", "Norm", "Abnorm", ">30"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            
            # Social Risk Factors with detailed descriptions
            social_risk_factors = st.slider("Social Risk Factors", 0, 4, 1)
            st.caption("**Social Risk Factor Descriptions:**")
            st.caption("0: No significant social risk factors")
            st.caption("1: Minor social challenges (e.g., occasional transportation issues)")
            st.caption("2: Moderate social challenges (e.g., housing instability, limited family support)")
            st.caption("3: Significant social challenges (e.g., homelessness, substance abuse, severe mental health issues)")
            st.caption("4: Severe social challenges with multiple compounding factors")
        
        with col_clinical2:
            blood_pressure_systolic = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120, step=1)
            blood_pressure_diastolic = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80, step=1)
            creatinine_level = st.number_input("Creatinine Level", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        
        # Additional Discharge and Stay Information
        st.subheader("🏠 Discharge & Previous Stay Information")
        col_discharge = st.columns(2)
        
        with col_discharge[0]:
            length_of_previous_stay = st.slider("Length of Previous Hospital Stay (days)", 0, 30, 5)
        
        with col_discharge[1]:
            discharged_to_home = st.checkbox("Discharged to Home", value=True)
            st.caption("*Checked: Yes, Unchecked: Skilled Nursing Facility or Other*")
        
        # Advanced Clinical Indicators
        st.subheader("🏥 Advanced Clinical Indicators")
        col_advanced1, col_advanced2 = st.columns(2)
        
        with col_advanced1:
            prior_icu_admissions = st.slider("Prior ICU Admissions", 0, 5, 0)
            prior_ed_visits_6m = st.slider("Prior ED Visits (Last 6 Months)", 0, 10, 0)
            polypharmacy_flag = st.checkbox("Polypharmacy Flag (≥5 or ≥10 medications)")
            high_risk_meds = st.checkbox("Taking High-Risk Medications")
            med_changes_during_stay = st.checkbox("Medication Changes During Stay")
        
        with col_advanced2:
            med_adherence_history = st.slider("Medication Adherence History (0-1 scale)", 0.0, 1.0, 0.8, step=0.01)
            new_med_started = st.checkbox("New Medication Started During Admission")
            icu_stay_during_admission = st.checkbox("ICU Stay During Current Admission")
            mechanical_ventilation = st.checkbox("Mechanical Ventilation Used")
            sepsis_diagnosis = st.checkbox("Sepsis Diagnosis")
        
        # Comorbidity Scores and Lab/Vital Indicators
        st.subheader("📊 Comorbidity Scores & Clinical Indicators")
        col_scores = st.columns(3)
        
        with col_scores[0]:
            charlson_comorbidity_index = st.slider("Charlson Comorbidity Index", 0, 10, 2)
        
        with col_scores[1]:
            elixhauser_comorbidity_score = st.slider("Elixhauser Comorbidity Score", 0, 15, 3)
        
        with col_scores[2]:
            abnormal_labs_flag = st.checkbox("Abnormal Lab Results Flag")
        
        col_vitals = st.columns(2)
        
        with col_vitals[0]:
            vital_instability_score = st.slider("Vital Instability Score (0-10)", 0.0, 10.0, 2.0, step=0.1)
        
        with col_vitals[1]:
            creatinine_change = st.slider("Change in Creatinine (mg/dL)", -2.0, 2.0, 0.0, step=0.1)
        
        # Behavioral & Engagement Signals
        st.subheader("🧠 Behavioral & Engagement Signals")
        col_behavioral = st.columns(2)
        
        with col_behavioral[0]:
            patient_portal_usage = st.radio("Patient Portal Usage", options=[0, 1, 2, 3], 
                                           format_func=lambda x: ["Never", "Rarely", "Sometimes", "Often"][x], index=2)
            prior_no_show_rate = st.slider("Prior No-Show Rate (0-1 scale)", 0.0, 1.0, 0.1, step=0.01)
            refused_medication = st.checkbox("Refused Medication")
            documented_non_compliance = st.checkbox("Documented Non-Compliance")
        
        with col_behavioral[1]:
            substance_use_disorder = st.checkbox("Substance Use Disorder History")
            depression_diagnosis = st.checkbox("Depression Diagnosis")
            anxiety_diagnosis = st.checkbox("Anxiety Diagnosis")
        
        # Diagnosis Representation Features
        st.subheader("📋 Diagnosis Representation")
        col_diagnosis = st.columns(2)
        
        with col_diagnosis[0]:
            ccs_diagnosis_category = st.slider("CCS Diagnosis Category", 0, 24, 5)
            chronic_condition_flag = st.checkbox("Chronic Condition Flag")
            high_risk_diagnosis = st.checkbox("High-Risk Diagnosis (CHF, COPD, Pneumonia)")
        
        with col_diagnosis[1]:
            count_chronic_conditions = st.slider("Count of Chronic Conditions", 0, 10, 2)
            drg_code = st.slider("DRG Code (1-999)", 1, 999, 100)
            principal_procedure_type = st.slider("Principal Procedure Type", 0, 49, 10)
        
        # Dynamic / Engineered Features
        st.subheader("⚙️ Dynamic / Engineered Features")
        col_dynamic1, col_dynamic2 = st.columns(2)
        
        with col_dynamic1:
            worsening_comorbidity_indicator = st.checkbox("Worsening Comorbidity Indicator")
        
        with col_dynamic2:
            admission_risk_score_percentile = st.slider("Admission Risk Score Percentile", 0, 100, 50)
            age_chf_interaction = st.slider("Age-CHF Interaction Term", 0.0, 100.0, 25.0, step=0.1)
            age_copd_interaction = st.slider("Age-COPD Interaction Term", 0.0, 100.0, 20.0, step=0.1)
        
        # Submit button
        submit_button = st.form_submit_button(
            label='🔍 Predict Readmission Risk', 
            help="Click to predict the patient's readmission risk",
            use_container_width=True
        )

# Model selection and prediction
with col2:
    st.markdown("<h2 class='sub-header'>📊 Prediction & Results</h2>", unsafe_allow_html=True)
    
    # Model selection
    model_option = st.selectbox(
        "Select Model for Prediction",
        ("Random Forest", "Logistic Regression"),
        help="Choose which machine learning model to use for prediction"
    )
    
    # Button to train models (if not already trained)
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        if st.button("⚙️ Train Models", use_container_width=True):
            with st.spinner("Training models... This may take a moment."):
                try:
                    # Initialize and train models
                    preprocessor = DataPreprocessor()
                    predictor = ReadmissionPredictor()
                    
                    # Load and preprocess data
                    df = preprocessor.load_data("data/diabetes.csv")
                    X, y = preprocessor.preprocess_data(df)
                    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
                    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
                    
                    # Train models
                    predictor.train_models(X_train_scaled, y_train)
                    
                    # Evaluate models
                    results = predictor.evaluate_models(X_test_scaled, y_test)
                    
                    # Store in session state
                    st.session_state.trained_predictor = predictor
                    st.session_state.preprocessor = preprocessor
                    st.session_state.model_trained = True
                    st.session_state.training_completed = True
                    st.session_state.results = results
                    st.session_state.X_train_scaled = X_train_scaled  # Store for SHAP explanations
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Show model performance
                    st.subheader("📈 Model Performance")
                    for model_name, metrics in results.items():
                        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                        
                        with col_perf1:
                            st.metric(label="Accuracy", value=f"{metrics['accuracy']:.3f}")
                        with col_perf2:
                            st.metric(label="Precision", value=f"{metrics['precision']:.3f}")
                        with col_perf3:
                            st.metric(label="Recall", value=f"{metrics['recall']:.3f}")
                        with col_perf4:
                            st.metric(label="F1-Score", value=f"{metrics['f1_score']:.3f}")
                            
                except Exception as e:
                    st.error(f"❌ Error training models: {str(e)}")
    
    with train_col2:
        if st.button("💾 Load Saved Models", use_container_width=True):
            # Try to load saved models if they exist
            model_files = {
                'Logistic Regression': 'models/logistic_regression_model.pkl',
                'Random Forest': 'models/random_forest_model.pkl'
            }
            
            try:
                predictor = ReadmissionPredictor()
                loaded_any = False
                
                for model_name, model_path in model_files.items():
                    if os.path.exists(model_path):
                        loaded_model = joblib.load(model_path)
                        predictor.trained_models[model_name] = loaded_model
                        loaded_any = True
                
                if loaded_any:
                    st.session_state.trained_predictor = predictor
                    st.session_state.model_trained = True
                    st.success("✅ Models loaded successfully!")
                else:
                    st.warning("⚠️ No saved models found. Please train the models first.")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
    
    # Display results when form is submitted
    if submit_button:
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the models first using the 'Train Models' button.")
        else:
            with st.spinner("Analyzing patient data..."):
                try:
                    # Prepare patient data for prediction
                    patient_data = {
                        'race': race,
                        'gender': gender,
                        'age': age_group,
                        'admission_type_id': admission_type_code,
                        'discharge_disposition_id': discharge_disposition_code,
                        'admission_source_id': admission_source_code,
                        'time_in_hospital': time_in_hospital,
                        'num_lab_procedures': num_lab_procedures,
                        'num_procedures': num_procedures,
                        'num_medications': num_medications,
                        'number_outpatient': number_outpatient,
                        'number_emergency': number_emergency,
                        'number_inpatient': number_inpatient,
                        'diag_1': diag_1_code,
                        'diag_2': diag_2_code,
                        'diag_3': diag_3_code,
                        'number_diagnoses': number_diagnoses,
                        'metformin': metformin,
                        'insulin': insulin,
                        'prior_admissions': prior_admissions,
                        'insurance_type': insurance_type,
                        'chf': int(chf),  # Convert boolean to int
                        'copd': int(copd),  # Convert boolean to int
                        'diabetes': int(diabetes),  # Convert boolean to int
                        'bp': int(bp),  # Convert boolean to int
                        'ckd': int(ckd),  # Convert boolean to int
                        'other_comorb': int(other_comorb),  # Convert boolean to int
                        # Enhanced Clinical Indicators
                        'hba1c_result': hba1c_result,
                        'blood_pressure_systolic': blood_pressure_systolic,
                        'blood_pressure_diastolic': blood_pressure_diastolic,
                        'bmi': bmi,
                        'creatinine_level': creatinine_level,
                        'length_of_previous_stay': length_of_previous_stay,
                        'discharged_to_home': int(discharged_to_home),  # Convert boolean to int
                        'social_risk_factors': social_risk_factors,
                        # Advanced Clinical Indicators
                        'prior_icu_admissions': prior_icu_admissions,
                        'prior_ed_visits_6m': prior_ed_visits_6m,
                        'polypharmacy_flag': int(polypharmacy_flag),
                        'high_risk_meds': int(high_risk_meds),
                        'med_changes_during_stay': int(med_changes_during_stay),
                        'med_adherence_history': med_adherence_history,
                        'new_med_started': int(new_med_started),
                        'icu_stay_during_admission': int(icu_stay_during_admission),
                        'mechanical_ventilation': int(mechanical_ventilation),
                        'sepsis_diagnosis': int(sepsis_diagnosis),
                        'charlson_comorbidity_index': charlson_comorbidity_index,
                        'elixhauser_comorbidity_score': elixhauser_comorbidity_score,
                        'abnormal_labs_flag': int(abnormal_labs_flag),
                        'vital_instability_score': vital_instability_score,
                        'creatinine_change': creatinine_change,
                        # Behavioral & Engagement Signals
                        'patient_portal_usage': patient_portal_usage,
                        'prior_no_show_rate': prior_no_show_rate,
                        'refused_medication': int(refused_medication),
                        'documented_non_compliance': int(documented_non_compliance),
                        'substance_use_disorder': int(substance_use_disorder),
                        'depression_diagnosis': int(depression_diagnosis),
                        'anxiety_diagnosis': int(anxiety_diagnosis),
                        # Diagnosis Representation Features
                        'ccs_diagnosis_category': ccs_diagnosis_category,
                        'chronic_condition_flag': int(chronic_condition_flag),
                        'high_risk_diagnosis': int(high_risk_diagnosis),
                        'count_chronic_conditions': count_chronic_conditions,
                        'drg_code': drg_code,
                        'principal_procedure_type': principal_procedure_type,
                        # Dynamic / Engineered Features
                        'worsening_comorbidity_indicator': int(worsening_comorbidity_indicator),
                        'admission_risk_score_percentile': admission_risk_score_percentile,
                        'age_chf_interaction': age_chf_interaction,
                        'age_copd_interaction': age_copd_interaction
                    }
                    
                    # Create a DataFrame with the patient data
                    patient_df = pd.DataFrame([patient_data])
                    
                    # Add a placeholder for the target variable (will be dropped later)
                    patient_df['readmitted_binary'] = 0
                    
                    # Use the same preprocessing steps as during training
                    preprocessor = st.session_state.preprocessor
                    
                    # Handle boolean columns by converting them to integers (True->1, False->0)
                    boolean_cols = patient_df.select_dtypes(include=['bool']).columns.tolist()
                    for col in boolean_cols:
                        patient_df[col] = patient_df[col].astype(int)
                    
                    # Encode categorical variables using the same encoders from training
                    categorical_cols = patient_df.select_dtypes(include=['object']).columns.tolist()
                    
                    for col in categorical_cols:
                        if col != 'readmitted_binary':  # Skip target variable
                            if col in preprocessor.label_encoders:
                                # Use the fitted encoder if available
                                try:
                                    patient_df[col] = preprocessor.label_encoders[col].transform(patient_df[col].astype(str))
                                except ValueError:
                                    # If new value not seen during training, use 0 as default
                                    patient_df[col] = 0
                            else:
                                # If the column wasn't in training data, assign 0 to all values
                                patient_df[col] = 0
                    
                    # Separate features from target
                    patient_features = patient_df.drop('readmitted_binary', axis=1)
                    
                    # Apply the same scaling as used during training
                    patient_scaled = preprocessor.scaler.transform(patient_features)
                    
                    # Ensure feature names match the training data
                    patient_scaled = pd.DataFrame(patient_scaled, columns=patient_features.columns)
                    
                    # Make prediction using the selected model
                    predictor = st.session_state.trained_predictor
                    # Ensure we're passing only the features that the model was trained on
                    prediction_result = predictor.predict_readmission_risk(model_option, patient_scaled.iloc[0].values)
                    
                    # Display results
                    st.subheader("📋 Prediction Results")
                    
                    if prediction_result['prediction'] == 1:
                        st.markdown(f"<div class='risk-high'>🚨 HIGH RISK: {prediction_result['risk_level']}</div>", 
                                  unsafe_allow_html=True)
                        st.error("⚠️ This patient has a high risk of being readmitted within 30 days.")
                    else:
                        st.markdown(f"<div class='risk-low'>✅ LOW RISK: {prediction_result['risk_level']}</div>", 
                                  unsafe_allow_html=True)
                        st.success("✅ This patient has a low risk of being readmitted within 30 days.")
                    
                    # Show probabilities
                    st.subheader("📊 Prediction Probabilities")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        st.metric(
                            label="Low Risk Probability",
                            value=f"{prediction_result['probability']['low_risk_prob']:.3f}",
                            delta=None
                        )
                    
                    with prob_col2:
                        st.metric(
                            label="High Risk Probability",
                            value=f"{prediction_result['probability']['high_risk_prob']:.3f}",
                            delta=None
                        )
                    
                    # Risk Score Breakdown (Transparent Scoring)
                    st.subheader("🧮 Risk Score Breakdown")
                    
                    # Get feature importance to show contribution breakdown
                    try:
                        feature_names = patient_scaled.columns.tolist()
                        importance_df = predictor.get_feature_importance(model_option, feature_names)
                        
                        # Get top contributing features for this patient
                        top_contributors = predictor.get_top_risk_factors(
                            model_option, 
                            feature_names, 
                            patient_scaled.iloc[0].values, 
                            top_n=5
                        )
                        
                        # Create a dataframe for the risk breakdown visualization
                        if model_option == 'Logistic Regression':
                            # For logistic regression, use the coefficient * patient value as contribution
                            contributions = top_contributors[['feature', 'contribution']].copy()
                            # Normalize contributions to percentage values (for display purposes)
                            total_abs_contrib = contributions['contribution'].sum()
                            if total_abs_contrib > 0:
                                contributions['percentage'] = (contributions['contribution'] / total_abs_contrib) * 100
                                contributions = contributions.sort_values('percentage', ascending=False)
                                
                                # Show the top contributors
                                breakdown_data = []
                                for _, row in contributions.iterrows():
                                    feature_name = row['feature']
                                    percentage = row['percentage']
                                    breakdown_data.append({
                                        'Factor': feature_name.replace('_', ' ').title(),
                                        'Contribution': f"+{percentage:.1f}%"
                                    })
                                
                                breakdown_df = pd.DataFrame(breakdown_data)
                                st.dataframe(breakdown_df, use_container_width=True)
                        else:
                            # For Random Forest, show feature importance
                            top_features = importance_df.head(5)
                            breakdown_data = []
                            total_importance = top_features['importance'].sum()
                            if total_importance > 0:
                                for _, row in top_features.iterrows():
                                    feature_name = row['feature']
                                    importance = row['importance']
                                    percentage = (importance / total_importance) * 100
                                    breakdown_data.append({
                                        'Factor': feature_name.replace('_', ' ').title(),
                                        'Contribution': f"+{percentage:.1f}%"
                                    })
                                
                                breakdown_df = pd.DataFrame(breakdown_data)
                                st.dataframe(breakdown_df, use_container_width=True)
                    
                    except Exception as e:
                        # Silently fail if risk score breakdown cannot be computed
                        pass
                    
                    # Feature Importance Chart
                    st.subheader("📈 Feature Importance")
                    try:
                        feature_names = patient_scaled.columns.tolist()
                        importance_df = predictor.get_feature_importance(model_option, feature_names)
                        
                        # Show top 10 most important features
                        top_features = importance_df.head(10)
                        
                        # Create a bar chart of feature importances
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Limit to top 10 features for readability
                        top_plot = top_features.head(10)
                        ax.barh(range(len(top_plot)), top_plot['importance'])
                        ax.set_yticks(range(len(top_plot)))
                        ax.set_yticklabels([name.replace('_', ' ').title() for name in top_plot['feature']])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 10 Most Important Features')
                        ax.invert_yaxis()  # Highest importance at top
                        
                        st.pyplot(fig)
                        plt.close()  # Close figure to free memory
                    
                    except Exception as e:
                        # Silently fail if feature importance chart cannot be generated
                        pass
                    
                    # Top Risk Factors Explanation
                    st.subheader("🔍 Top Risk Factors for This Patient")
                    try:
                        feature_names = patient_scaled.columns.tolist()
                        top_factors = predictor.get_top_risk_factors(
                            model_option, 
                            feature_names, 
                            patient_scaled.iloc[0].values, 
                            top_n=5
                        )
                        
                        if not top_factors.empty:
                            # Create a more detailed explanation
                            st.markdown("### 🎯 This patient is high risk mainly due to:")
                            
                            factor_descriptions = {
                                'time_in_hospital': f"{time_in_hospital} days hospital stay",
                                'num_medications': f"{num_medications} medications (polypharmacy)",
                                'number_emergency': f"{number_emergency} emergency visits",
                                'prior_admissions': f"{prior_admissions} prior admissions",
                                'number_diagnoses': f"{number_diagnoses} diagnoses",
                                'chf': "Congestive heart failure diagnosis",
                                'copd': "COPD diagnosis",
                                'diabetes': "Diabetes diagnosis",
                                'bp': "Hypertension diagnosis",
                                'ckd': "Chronic kidney disease diagnosis"
                            }
                            
                            for i, (_, row) in enumerate(top_factors.iterrows(), 1):
                                feature = row['feature']
                                value = row['value']
                                
                                desc_key = feature.lower()
                                if desc_key in factor_descriptions:
                                    description = factor_descriptions[desc_key]
                                    st.write(f"{i}. {description}")
                                else:
                                    st.write(f"{i}. {feature.replace('_', ' ').title()} (value: {value})")
                        else:
                            # No top risk factors could be determined
                            pass
                    
                    except Exception as e:
                        # Silently fail if top risk factors cannot be computed
                        pass
                    
                    # Clinical Recommendation Engine
                    st.subheader("🩺 Clinical Recommendations")
                    if prediction_result['prediction'] == 1:  # High risk
                        st.markdown("""
                        <div class="warning-box">
                        <strong>🔴 HIGH RISK PATIENT - IMMEDIATE ACTIONS RECOMMENDED:</strong>
                        <ul>
                            <li>📅 Schedule follow-up appointment within <strong>7 days</strong></li>
                            <li>💊 Conduct medication reconciliation review</li>
                            <li>🩺 Arrange diabetes education referral (if applicable)</li>
                            <li>👨‍⚕️ Assign care coordination nurse</li>
                            <li>📞 Provide discharge instructions with clear follow-up plan</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # Low risk
                        st.markdown("""
                        <div class="info-box">
                        <strong>🟢 LOWER RISK PATIENT - STANDARD FOLLOW-UP:</strong>
                        <ul>
                            <li>📅 Schedule follow-up appointment within <strong>14-30 days</strong></li>
                            <li>💊 Review medications at next visit</li>
                            <li>🏥 Monitor condition as per standard protocol</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # SHAP Explanation (if model supports it)
                    if model_option == 'Random Forest':  # SHAP works better with tree models
                        st.subheader("🔬 SHAP Explanation")
                        try:
                            # Use the stored training data for SHAP
                            shap_values, expected_value = predictor.get_shap_explanation(
                                model_option, 
                                patient_scaled.iloc[0].values
                            )
                            
                            # Get feature names
                            feature_names = patient_scaled.columns.tolist()
                            
                            # Create SHAP explanation object
                            explainer = shap.TreeExplainer(predictor.trained_models[model_option])
                            
                            # Get SHAP values for this specific patient
                            shap_values_patient = explainer.shap_values(patient_scaled.iloc[0].values.reshape(1, -1))
                            
                            # For binary classification with 2 classes, take the positive class (index 1)
                            if isinstance(shap_values_patient, list) and len(shap_values_patient) == 2:
                                shap_values_single = shap_values_patient[1][0]  # Positive class for this patient
                            elif len(shap_values_patient.shape) > 2:  # Multiple outputs
                                shap_values_single = shap_values_patient[0, :, 1] if shap_values_patient.shape[2] > 1 else shap_values_patient[0, :]
                            else:
                                shap_values_single = shap_values_patient[0, :]  # For single output
                            
                            # Create a DataFrame for visualization
                            shap_df = pd.DataFrame({
                                'feature': feature_names,
                                'shap_value': shap_values_single
                            }).sort_values(by='shap_value', key=abs, ascending=False)
                            
                            # Show top 10 SHAP values
                            top_shap = shap_df.head(10)
                            
                            import matplotlib.pyplot as plt
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            colors = ['red' if x > 0 else 'blue' for x in top_shap['shap_value']]
                            ax.barh(range(len(top_shap)), top_shap['shap_value'], color=colors)
                            ax.set_yticks(range(len(top_shap)))
                            ax.set_yticklabels([name.replace('_', ' ').title() for name in top_shap['feature']])
                            ax.set_xlabel('SHAP Value')
                            ax.set_title('SHAP Feature Contributions (Red increases risk, Blue decreases risk)')
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            
                            st.pyplot(fig)
                            plt.close()
                            
                        except Exception as e:
                            # Silently fail if SHAP cannot be generated, without showing warning
                            pass  # Do nothing if SHAP fails
                    
                    # Risk factors analysis
                    st.subheader("🔍 Additional Risk Factor Analysis")
                    
                    # Create a summary of patient data that contributes to risk
                    risk_indicators = []
                    
                    if time_in_hospital > 7:
                        risk_indicators.append(f"Extended hospital stay ({time_in_hospital} days)")
                    if num_medications > 15:
                        risk_indicators.append(f"High medication count ({num_medications})")
                    if number_diagnoses > 8:
                        risk_indicators.append(f"Multiple diagnoses ({number_diagnoses})")
                    if number_inpatient > 2:
                        risk_indicators.append(f"Frequent inpatient visits ({number_inpatient})")
                    if number_emergency > 3:
                        risk_indicators.append(f"Frequent emergency visits ({number_emergency})")
                    # NEW FEATURES RISK INDICATORS
                    if hba1c_result in ['Abnorm', '>30']:
                        risk_indicators.append(f"Abnormal HbA1c result ({hba1c_result})")
                    if blood_pressure_systolic > 160 or blood_pressure_diastolic > 100:
                        risk_indicators.append(f"High blood pressure ({blood_pressure_systolic}/{blood_pressure_diastolic})")
                    if bmi > 35:
                        risk_indicators.append(f"High BMI ({bmi})")
                    if creatinine_level > 2.0:
                        risk_indicators.append(f"Elevated creatinine level ({creatinine_level})")
                    if length_of_previous_stay > 10:
                        risk_indicators.append(f"Long previous hospital stay ({length_of_previous_stay} days)")
                    if not discharged_to_home:
                        risk_indicators.append("Discharged to skilled nursing facility or other")
                    if social_risk_factors > 2:
                        risk_indicators.append(f"High social risk factors ({social_risk_factors}/4)")
                    # ADVANCED CLINICAL FEATURES RISK INDICATORS
                    if prior_icu_admissions > 0:
                        risk_indicators.append(f"Prior ICU admissions ({prior_icu_admissions})")
                    if prior_ed_visits_6m > 2:
                        risk_indicators.append(f"Multiple ED visits in past 6 months ({prior_ed_visits_6m})")
                    if polypharmacy_flag:
                        risk_indicators.append("Polypharmacy flag (≥5 or ≥10 medications)")
                    if high_risk_meds:
                        risk_indicators.append("Taking high-risk medications")
                    if med_changes_during_stay:
                        risk_indicators.append("Medication changes during stay")
                    if med_adherence_history < 0.5:
                        risk_indicators.append(f"Poor medication adherence ({med_adherence_history:.2f})")
                    if new_med_started:
                        risk_indicators.append("New medication started during admission")
                    if icu_stay_during_admission:
                        risk_indicators.append("ICU stay during current admission")
                    if mechanical_ventilation:
                        risk_indicators.append("Mechanical ventilation used")
                    if sepsis_diagnosis:
                        risk_indicators.append("Sepsis diagnosis")
                    if charlson_comorbidity_index > 5:
                        risk_indicators.append(f"High Charlson Comorbidity Index ({charlson_comorbidity_index})")
                    if elixhauser_comorbidity_score > 8:
                        risk_indicators.append(f"High Elixhauser Comorbidity Score ({elixhauser_comorbidity_score})")
                    if abnormal_labs_flag:
                        risk_indicators.append("Abnormal lab results flag")
                    if vital_instability_score > 6:
                        risk_indicators.append(f"High vital instability score ({vital_instability_score:.1f})")
                    if abs(creatinine_change) > 0.5:
                        risk_indicators.append(f"Significant change in creatinine ({creatinine_change:.2f})")
                    # BEHAVIORAL & ENGAGEMENT SIGNALS RISK INDICATORS
                    if patient_portal_usage < 2:
                        risk_indicators.append(f"Low patient portal usage ({['Never', 'Rarely', 'Sometimes', 'Often'][patient_portal_usage]})")
                    if prior_no_show_rate > 0.3:
                        risk_indicators.append(f"High no-show rate ({prior_no_show_rate:.2f})")
                    if refused_medication:
                        risk_indicators.append("Refused medication")
                    if documented_non_compliance:
                        risk_indicators.append("Documented non-compliance")
                    if substance_use_disorder:
                        risk_indicators.append("Substance use disorder history")
                    if depression_diagnosis:
                        risk_indicators.append("Depression diagnosis")
                    if anxiety_diagnosis:
                        risk_indicators.append("Anxiety diagnosis")
                    # DIAGNOSIS REPRESENTATION FEATURES RISK INDICATORS
                    if chronic_condition_flag:
                        risk_indicators.append("Chronic condition flag")
                    if high_risk_diagnosis:
                        risk_indicators.append("High-risk diagnosis (CHF, COPD, Pneumonia)")
                    if count_chronic_conditions > 4:
                        risk_indicators.append(f"High count of chronic conditions ({count_chronic_conditions})")
                    # DYNAMIC / ENGINEERED FEATURES RISK INDICATORS
                    if worsening_comorbidity_indicator:
                        risk_indicators.append("Worsening comorbidity indicator")
                    if admission_risk_score_percentile > 75:
                        risk_indicators.append(f"High admission risk score percentile ({admission_risk_score_percentile:.1f})")
                    if age_chf_interaction > 50:
                        risk_indicators.append(f"High age-CHF interaction ({age_chf_interaction:.1f})")
                    if age_copd_interaction > 50:
                        risk_indicators.append(f"High age-COPD interaction ({age_copd_interaction:.1f})")
                    
                    if risk_indicators:
                        st.warning("⚠️ Additional risk factors detected:")
                        for indicator in risk_indicators:
                            st.write(f"- {indicator}")
                    else:
                        st.info("ℹ️ No significant additional risk factors identified based on entered data.")
                        
                    # Show patient summary
                    st.subheader("📋 Patient Summary")
                    patient_summary = {
                        "Age Group": age_group,
                        "Gender": gender,
                        "Race": race,
                        "Time in Hospital": f"{time_in_hospital} days",
                        "Number of Medications": num_medications,
                        "Number of Diagnoses": number_diagnoses,
                        "Lab Procedures": num_lab_procedures,
                        "Prior Admissions (12 mo)": str(prior_admissions),
                        "Insurance Type": insurance_type,
                        "CHF": "Yes" if chf else "No",
                        "COPD": "Yes" if copd else "No",
                        "Diabetes": "Yes" if diabetes else "No",
                        "Hypertension": "Yes" if bp else "No",
                        "CKD": "Yes" if ckd else "No",
                        "Other Comorbidities": "Yes" if other_comorb else "No",
                        # Enhanced Clinical Indicators
                        "HbA1c Result": hba1c_result,
                        "BMI": f"{bmi:.1f}",
                        "Systolic BP": f"{blood_pressure_systolic} mmHg",
                        "Diastolic BP": f"{blood_pressure_diastolic} mmHg",
                        "Creatinine Level": f"{creatinine_level:.2f} mg/dL",
                        "Length of Previous Stay": f"{length_of_previous_stay} days",
                        "Discharged to Home": "Yes" if discharged_to_home else "No",
                        "Social Risk Factors": social_risk_factors,
                        # Advanced Clinical Indicators
                        "Prior ICU Admissions": prior_icu_admissions,
                        "Prior ED Visits (Last 6 Mo)": prior_ed_visits_6m,
                        "Polypharmacy Flag": "Yes" if polypharmacy_flag else "No",
                        "High-Risk Meds": "Yes" if high_risk_meds else "No",
                        "Med Changes During Stay": "Yes" if med_changes_during_stay else "No",
                        "Med Adherence History": f"{med_adherence_history:.2f}",
                        "New Med Started": "Yes" if new_med_started else "No",
                        "ICU Stay During Admission": "Yes" if icu_stay_during_admission else "No",
                        "Mechanical Ventilation": "Yes" if mechanical_ventilation else "No",
                        "Sepsis Diagnosis": "Yes" if sepsis_diagnosis else "No",
                        "Charlson Comorbidity Index": charlson_comorbidity_index,
                        "Elixhauser Comorbidity Score": elixhauser_comorbidity_score,
                        "Abnormal Labs Flag": "Yes" if abnormal_labs_flag else "No",
                        "Vital Instability Score": f"{vital_instability_score:.1f}",
                        "Creatinine Change": f"{creatinine_change:.2f}",
                        # Behavioral & Engagement Signals
                        "Patient Portal Usage": ["Never", "Rarely", "Sometimes", "Often"][patient_portal_usage],
                        "Prior No-Show Rate": f"{prior_no_show_rate:.2f}",
                        "Refused Medication": "Yes" if refused_medication else "No",
                        "Documented Non-Compliance": "Yes" if documented_non_compliance else "No",
                        "Substance Use Disorder": "Yes" if substance_use_disorder else "No",
                        "Depression Diagnosis": "Yes" if depression_diagnosis else "No",
                        "Anxiety Diagnosis": "Yes" if anxiety_diagnosis else "No",
                        # Diagnosis Representation Features
                        "CCS Diagnosis Category": ccs_diagnosis_category,
                        "Chronic Condition Flag": "Yes" if chronic_condition_flag else "No",
                        "High-Risk Diagnosis": "Yes" if high_risk_diagnosis else "No",
                        "Count of Chronic Conditions": count_chronic_conditions,
                        "DRG Code": drg_code,
                        "Principal Procedure Type": principal_procedure_type,
                        # Dynamic / Engineered Features
                        "Worsening Comorbidity Indicator": "Yes" if worsening_comorbidity_indicator else "No",
                        "Admission Risk Score Percentile": f"{admission_risk_score_percentile:.1f}",
                        "Age-CHF Interaction": f"{age_chf_interaction:.1f}",
                        "Age-COPD Interaction": f"{age_copd_interaction:.1f}"
                    }
                    
                    patient_summary_df = pd.DataFrame(list(patient_summary.items()), 
                                                   columns=['Attribute', 'Value'])
                    st.dataframe(patient_summary_df, use_container_width=True)
                    
                    st.info("ℹ️ Note: All entered features are now incorporated into the prediction model for improved accuracy.")
                    
                    # Download Risk Assessment Report
                    st.subheader("📄 Patient Risk Assessment Report")
                    
                    if 'trained_predictor' in st.session_state and st.session_state.model_trained:
                        import datetime
                        from io import BytesIO
                        
                        # Prepare report data
                        report_data = {
                            'patient_summary': patient_summary,
                            'risk_level': prediction_result['risk_level'],
                            'high_risk_probability': prediction_result['probability']['high_risk_prob'],
                            'low_risk_probability': prediction_result['probability']['low_risk_prob'],
                            'top_risk_factors': risk_indicators[:5] if risk_indicators else ["No significant risk factors identified"],
                            'clinical_recommendations': "High-risk patient: Schedule follow-up within 7 days, conduct medication reconciliation, arrange diabetes education referral if applicable, assign care coordination nurse." if prediction_result['prediction'] == 1 else "Lower-risk patient: Schedule follow-up within 14-30 days, review medications at next visit, monitor condition as per standard protocol.",
                            'date_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'model_version': "Hospital Readmission Prediction v2.0"
                        }
                        
                        # Create a simple text-based report
                        import reportlab
                        from reportlab.lib.pagesizes import letter
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                        from reportlab.lib.units import inch
                        from reportlab.lib import colors
                        
                        def create_pdf_report(report_data):
                            buffer = BytesIO()
                            doc = SimpleDocTemplate(buffer, pagesize=letter)
                            styles = getSampleStyleSheet()
                            story = []
                            
                            # Title
                            title_style = ParagraphStyle(
                                'CustomTitle',
                                parent=styles['Heading1'],
                                fontSize=18,
                                spaceAfter=30,
                                alignment=1  # Center alignment
                            )
                            title = Paragraph("Hospital Readmission Risk Assessment Report", title_style)
                            story.append(title)
                            
                            # Date and time
                            date_para = Paragraph(f"Assessment Date: {report_data['date_time']}", styles['Normal'])
                            story.append(date_para)
                            story.append(Spacer(1, 12))
                            
                            # Patient Summary
                            story.append(Paragraph("Patient Information:", styles['Heading2']))
                            patient_data = [[key, str(value)] for key, value in report_data['patient_summary'].items()]
                            patient_table = Table(patient_data)
                            patient_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (0, -1), 8),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(patient_table)
                            story.append(Spacer(1, 20))
                            
                            # Risk Assessment
                            story.append(Paragraph("Risk Assessment:", styles['Heading2']))
                            risk_data = [
                                ["Risk Level", report_data['risk_level']],
                                ["High Risk Probability", f"{report_data['high_risk_probability']:.3f}"],
                                ["Low Risk Probability", f"{report_data['low_risk_probability']:.3f}"]
                            ]
                            risk_table = Table(risk_data)
                            risk_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (0, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(risk_table)
                            story.append(Spacer(1, 20))
                            
                            # Contributing Factors
                            story.append(Paragraph("Contributing Risk Factors:", styles['Heading2']))
                            factors_data = [["Factor", "Description"]]
                            for factor in report_data['top_risk_factors']:
                                factors_data.append(["Risk Factor", factor])
                            factors_table = Table(factors_data)
                            factors_table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(factors_table)
                            story.append(Spacer(1, 20))
                            
                            # Recommended Actions
                            story.append(Paragraph("Recommended Actions:", styles['Heading2']))
                            rec_para = Paragraph(report_data['clinical_recommendations'], styles['Normal'])
                            story.append(rec_para)
                            story.append(Spacer(1, 20))
                            
                            # Model Information
                            story.append(Paragraph(f"Model Version: {report_data['model_version']}", styles['Normal']))
                            
                            doc.build(story)
                            buffer.seek(0)
                            return buffer
                        
                        # Create and offer PDF download
                        pdf_buffer = create_pdf_report(report_data)
                        
                        st.download_button(
                            label="📄 Download Risk Assessment Report (PDF)",
                            data=pdf_buffer,
                            file_name=f"readmission_risk_assessment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.info("Train the models first to enable report generation.")
                        
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
                    st.error("Please make sure models are trained and all fields are filled correctly.")

# Training results section if models were trained
if st.session_state.training_completed and hasattr(st.session_state, 'results'):
    st.markdown("---")
    st.markdown("<h2 class='sub-header'>📈 Training Results</h2>", unsafe_allow_html=True)
    
    # Show model comparison
    results = st.session_state.results
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.subheader("Model Comparison")
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    with col_comp2:
        st.subheader("Best Performing Model")
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        
        st.metric(
            label="Best Model",
            value=best_model,
            delta=f"Accuracy: {best_accuracy:.3f}"
        )
        
        st.info(f"The {best_model} model achieved the highest accuracy of {best_accuracy:.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>Hospital Readmission Risk Predictor</strong> | Advanced Machine Learning Platform</p>
    <p>Powered by cutting-edge algorithms to improve patient outcomes and reduce readmission rates</p>
</div>
""", unsafe_allow_html=True)