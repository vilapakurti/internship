import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from utils.data_preprocessing import DataPreprocessor
from models.model_training import ReadmissionPredictor
import pickle

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
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4472c4;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #d13838;
        font-weight: bold;
        font-size: 1.2em;
    }
    .risk-low {
        color: #76b041;
        font-weight: bold;
        font-size: 1.2em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🏥 Hospital Readmission Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# Sidebar
with st.sidebar:
    st.header("About This Application")
    st.info("""
    This application predicts the risk of 30-day hospital readmission based on patient information.
    
    **How it works:**
    1. Enter patient details in the input fields
    2. Select the model to use for prediction
    3. Click 'Predict' to get the risk assessment
    """)
    
    st.header("Model Information")
    st.success("**Algorithms Used:**")
    st.write("- Logistic Regression")
    st.write("- Random Forest")
    
    st.header("Risk Levels")
    st.write("**High Risk:** Patient likely to be readmitted within 30 days")
    st.write("**Low Risk:** Patient unlikely to be readmitted within 30 days")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h2 class='sub-header'>Patient Information</h2>", unsafe_allow_html=True)
    
    # Create input fields for patient data
    with st.form(key='patient_form'):
        # Demographics
        st.subheader("Demographics")
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
        st.subheader("Admission Details")
        col_adm1, col_adm2 = st.columns(2)
        
        with col_adm1:
            admission_type = st.selectbox("Admission Type", list(range(1, 9)))
            admission_source = st.selectbox("Admission Source", list(range(1, 20)))
        
        with col_adm2:
            time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
            number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)
        
        # Medical Information
        st.subheader("Medical Details")
        col_med1, col_med2 = st.columns(2)
        
        with col_med1:
            num_medications = st.slider("Number of Medications", 1, 25, 10)
            num_lab_procedures = st.slider("Number of Lab Procedures", 1, 50, 15)
            num_procedures = st.slider("Number of Procedures", 0, 5, 1)
        
        with col_med2:
            number_outpatient = st.slider("Outpatient Visits", 0, 10, 2)
            number_emergency = st.slider("Emergency Visits", 0, 10, 1)
            number_inpatient = st.slider("Inpatient Visits", 0, 10, 1)
        
        # Medications
        st.subheader("Medication Status")
        col_med_status = st.columns(3)
        
        with col_med_status[0]:
            metformin = st.selectbox("Metformin", ["No", "Up", "Down", "Steady"])
        with col_med_status[1]:
            insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"])
        with col_med_status[2]:
            change = st.selectbox("Change in Medication", ["No", "Ch"])
        
        # Submit button
        submit_button = st.form_submit_button(label='🔍 Predict Readmission Risk', 
                                             help="Click to predict the patient's readmission risk")

# Model selection and prediction
with col2:
    st.markdown("<h2 class='sub-header'>Prediction & Results</h2>", unsafe_allow_html=True)
    
    # Model selection
    model_option = st.selectbox(
        "Select Model for Prediction",
        ("Random Forest", "Logistic Regression"),
        help="Choose which machine learning model to use for prediction"
    )
    
    # Button to train models (if not already trained)
    if st.button("⚙️ Train Models"):
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
                st.session_state.trained_models = predictor
                st.session_state.preprocessor = preprocessor
                st.session_state.model_trained = True
                
                st.success("✅ Models trained successfully!")
                
                # Show model performance
                st.subheader("Model Performance")
                for model_name, metrics in results.items():
                    st.metric(
                        label=f"{model_name}",
                        value=f"{metrics['accuracy']:.3f}",
                        delta=f"Precision: {metrics['precision']:.3f}"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
    
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
                        'admission_type_id': admission_type,
                        'admission_source_id': admission_source,
                        'time_in_hospital': time_in_hospital,
                        'num_lab_procedures': num_lab_procedures,
                        'num_procedures': num_procedures,
                        'num_medications': num_medications,
                        'number_outpatient': number_outpatient,
                        'number_emergency': number_emergency,
                        'number_inpatient': number_inpatient,
                        'number_diagnoses': number_diagnoses,
                        'metformin': metformin,
                        'insulin': insulin,
                        'change': change
                    }
                    
                    # Create a DataFrame with the patient data
                    patient_df = pd.DataFrame([patient_data])
                    
                    # Add the target column temporarily (will be removed in preprocessing)
                    patient_df['readmitted_binary'] = 0
                    
                    # Preprocess the patient data the same way as training data
                    # We'll manually encode the categorical variables to match our training process
                    preprocessor = st.session_state.preprocessor
                    
                    # Encode categorical variables
                    categorical_cols = ['race', 'gender', 'age', 'metformin', 'insulin', 'change']
                    
                    for col in categorical_cols:
                        if col in preprocessor.label_encoders:
                            # Use the fitted encoder if available
                            try:
                                patient_df[col] = preprocessor.label_encoders[col].transform(patient_df[col].astype(str))
                            except ValueError:
                                # If new value not seen during training, use 0 as default
                                patient_df[col] = 0
                        else:
                            # Create and fit an encoder for this column
                            le = preprocessor.label_encoders.get(col, None)
                            if le is None:
                                le = preprocessor.label_encoders[col] = pd.Series(patient_df[col]).astype(str).drop_duplicates().reset_index(drop=True)
                                le = pd.Series(range(len(le)), index=le.values)
                                patient_df[col] = patient_df[col].map(le).fillna(0)
                    
                    # Remove the target column
                    patient_features = patient_df.drop('readmitted_binary', axis=1)
                    
                    # Scale the features
                    patient_scaled = preprocessor.scaler.transform(patient_features)
                    
                    # Make prediction
                    predictor = st.session_state.trained_models
                    prediction_result = predictor.predict_readmission_risk(model_option, patient_scaled[0])
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    if prediction_result['prediction'] == 1:
                        st.markdown(f"<div class='risk-high'>🚨 HIGH RISK: {prediction_result['risk_level']}</div>", 
                                  unsafe_allow_html=True)
                        st.error("⚠️ This patient has a high risk of being readmitted within 30 days.")
                    else:
                        st.markdown(f"<div class='risk-low'>✅ LOW RISK: {prediction_result['risk_level']}</div>", 
                                  unsafe_allow_html=True)
                        st.success("✅ This patient has a low risk of being readmitted within 30 days.")
                    
                    # Show probabilities
                    st.subheader("Prediction Probabilities")
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
                    
                    # Risk factors
                    st.subheader("Key Risk Factors")
                    risk_factors = []
                    
                    if time_in_hospital > 7:
                        risk_factors.append(f"Long hospital stay ({time_in_hospital} days)")
                    if num_medications > 15:
                        risk_factors.append(f"High number of medications ({num_medications})")
                    if number_diagnoses > 8:
                        risk_factors.append(f"High number of diagnoses ({number_diagnoses})")
                    if number_inpatient > 2:
                        risk_factors.append(f"Multiple inpatient visits ({number_inpatient})")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(f"⚠️ {factor}")
                    else:
                        st.info("ℹ️ No significant risk factors identified")
                        
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p><strong>Hospital Readmission Risk Predictor</strong> | 
    Machine Learning Model for Predicting 30-Day Hospital Readmission</p>
    <p>⚠️ This tool is for educational purposes only and should not be used for actual medical decisions.</p>
</div>
""", unsafe_allow_html=True)