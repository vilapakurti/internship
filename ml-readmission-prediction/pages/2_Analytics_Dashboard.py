import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add the project directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from utils.data_preprocessing import DataPreprocessor
from models.model_training import ReadmissionPredictor

# Set page configuration
st.set_page_config(
    page_title="Hospital Analytics Dashboard",
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🏥 Hospital Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Hospital Management Dashboard for Readmission Analytics</p>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = None

# Sidebar
with st.sidebar:
    st.header("ℹ️ Dashboard Information")
    st.info("""
    This dashboard provides hospital-level analytics for readmission management.
    
    **Metrics Tracked:**
    - Overall readmission rates
    - High-risk patient percentages
    - Common diagnoses
    - Average length of stay
    """)
    
    st.header("⚙️ Data Controls")
    refresh_data = st.button("🔄 Refresh Analytics Data")
    
    if refresh_data or st.session_state.analytics_data is None:
        with st.spinner("Loading analytics data..."):
            try:
                preprocessor = DataPreprocessor()
                df = preprocessor.load_data("data/diabetes.csv")
                
                # Process the data to extract analytics
                processed_df = df.copy()
                
                # Convert readmission to binary for analytics
                processed_df['readmitted_binary'] = processed_df['readmitted'].apply(
                    lambda x: 1 if x == '<30' else 0
                )
                
                st.session_state.analytics_data = processed_df
                st.success("Analytics data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

# Add New Patient Data Section
st.subheader("➕ Add New Patient Data")
with st.expander("Enter New Patient Information"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "?"], index=0)
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown/Invalid"], index=0)
        age = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=5)
        admission_type_id = st.selectbox("Admission Type", list(range(1, 9)), index=0)
        admission_source_id = st.selectbox("Admission Source", list(range(1, 20)), index=0)
        discharge_disposition_id = st.selectbox("Discharge Disposition", list(range(1, 30)), index=0)
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 5)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 50, 15)
        num_procedures = st.slider("Number of Procedures", 0, 5, 1)
    
    with col2:
        num_medications = st.slider("Number of Medications", 1, 25, 10)
        number_outpatient = st.slider("Outpatient Visits", 0, 10, 2)
        number_emergency = st.slider("Emergency Visits", 0, 10, 1)
        number_inpatient = st.slider("Inpatient Visits", 0, 10, 1)
        diag_1 = st.selectbox("Primary Diagnosis", ['250', '428', '414', '403', '276', '427', '584', '250.01', '?'], index=0)
        diag_2 = st.selectbox("Secondary Diagnosis", ['250', '428', '414', '403', '276', '427', '584', '250.01', '?'], index=0)
        diag_3 = st.selectbox("Additional Diagnosis", ['250', '428', '414', '403', '276', '427', '584', '250.01', '?'], index=0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)
        metformin = st.selectbox("Metformin", ["No", "Up", "Down", "Steady"], index=0)
        insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"], index=0)
    
    with col3:
        prior_admissions = st.slider("Prior Admissions (Last 12 Months)", 0, 20, 0)
        insurance_type = st.selectbox("Insurance Type", ["Private", "Medicaid", "Medicare", "Self-Pay", "Government", "Other"], index=0)
        chf = st.checkbox("Congestive Heart Failure (CHF)")
        copd = st.checkbox("Chronic Obstructive Pulmonary Disease (COPD)")
        diabetes = st.checkbox("Diabetes")
        bp = st.checkbox("Hypertension (BP)")
        ckd = st.checkbox("Chronic Kidney Disease (CKD)")
        other_comorb = st.checkbox("Other Comorbidity")
        # New enhanced features
        hba1c_result = st.selectbox("HbA1c Result", ["None", "Norm", "Abnorm", ">30"], index=0)
        blood_pressure_systolic = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120, step=1)
        blood_pressure_diastolic = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150, value=80, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        creatinine_level = st.number_input("Creatinine Level", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        length_of_previous_stay = st.slider("Length of Previous Hospital Stay (days)", 0, 30, 5)
        discharged_to_home = st.checkbox("Discharged to Home", value=True)
        social_risk_factors = st.slider("Social Risk Factors", 0, 4, 1)
    
    # Readmission outcome
    readmitted = st.radio("Readmitted within 30 days?", ["No", "Yes"], index=0)
    
    # Add patient button
    if st.button("➕ Add Patient to Database"):
        if st.session_state.analytics_data is not None:
            # Create new patient record
            new_patient = {
                'race': race,
                'gender': gender,
                'age': age,
                'admission_type_id': admission_type_id,
                'admission_source_id': admission_source_id,
                'discharge_disposition_id': discharge_disposition_id,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'diag_1': diag_1,
                'diag_2': diag_2,
                'diag_3': diag_3,
                'number_diagnoses': number_diagnoses,
                'metformin': metformin,
                'insulin': insulin,
                'prior_admissions': prior_admissions,
                'insurance_type': insurance_type,
                'chf': chf,
                'copd': copd,
                'diabetes': diabetes,
                'bp': bp,
                'ckd': ckd,
                'other_comorb': other_comorb,
                # Enhanced features
                'hba1c_result': hba1c_result,
                'blood_pressure_systolic': blood_pressure_systolic,
                'blood_pressure_diastolic': blood_pressure_diastolic,
                'bmi': bmi,
                'creatinine_level': creatinine_level,
                'length_of_previous_stay': length_of_previous_stay,
                'discharged_to_home': discharged_to_home,
                'social_risk_factors': social_risk_factors,
                # Outcome
                'readmitted': '<30' if readmitted == 'Yes' else '>30',
                'readmitted_binary': 1 if readmitted == 'Yes' else 0
            }
            
            # Add to analytics data
            new_df = pd.DataFrame([new_patient])
            st.session_state.analytics_data = pd.concat([st.session_state.analytics_data, new_df], ignore_index=True)
            
            st.success("✅ Patient data added successfully! Analytics will update automatically.")
            
            # Option to save updated data back to file
            if st.button("💾 Save Updated Data to File"):
                try:
                    # Save the updated data back to the CSV file
                    st.session_state.analytics_data.to_csv("data/diabetes.csv", index=False)
                    st.success("💾 Updated patient data saved to data/diabetes.csv")
                except Exception as e:
                    st.error(f"❌ Error saving data to file: {str(e)}")
        else:
            st.error("Please load the analytics data first using the sidebar controls.")

# Main content
if st.session_state.analytics_data is not None:
    df = st.session_state.analytics_data
    
    # Calculate key metrics
    total_patients = len(df)
    readmitted_patients = df['readmitted_binary'].sum()
    readmission_rate = (readmitted_patients / total_patients) * 100 if total_patients > 0 else 0
    
    # Assuming high risk means readmitted within 30 days
    high_risk_percentage = readmission_rate
    
    # Average hospital stay
    avg_length_of_stay = df['time_in_hospital'].mean()
    
    # Most common diagnoses
    # Count occurrences of each diagnosis across all diagnosis columns
    diag_counts = {}
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            counts = df[col].value_counts()
            for diag, count in counts.items():
                if diag in diag_counts:
                    diag_counts[diag] += count
                else:
                    diag_counts[diag] = count
    
    # Sort by count to find most common
    sorted_diagnoses = sorted(diag_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_diagnosis = sorted_diagnoses[0][0] if sorted_diagnoses else "N/A"
    most_common_diagnosis_count = sorted_diagnoses[0][1] if sorted_diagnoses else 0
    
    # Create metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{readmission_rate:.2f}%</div>
            <div class="metric-label">Overall Readmission Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{high_risk_percentage:.2f}%</div>
            <div class="metric-label">High-Risk Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{most_common_diagnosis}</div>
            <div class="metric-label">Most Common Diagnosis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_length_of_stay:.1f} days</div>
            <div class="metric-label">Avg. Hospital Stay</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts and visualizations
    st.subheader("📈 Detailed Analytics")
    
    # Row 1: Readmission Distribution and Length of Stay
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### Readmission Distribution")
        readmission_counts = df['readmitted_binary'].value_counts()
        readmission_labels = ['Not Readmitted (>30 days)', 'Readmitted (<30 days)']
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        colors = ['#76b041', '#d13838']
        wedges, texts, autotexts = ax1.pie(
            readmission_counts.values, 
            labels=readmission_labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        ax1.axis('equal')
        st.pyplot(fig1)
        plt.close()
    
    with chart_col2:
        st.markdown("### Length of Stay Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(df['time_in_hospital'], bins=20, color='#4472c4', edgecolor='black')
        ax2.set_xlabel('Days in Hospital')
        ax2.set_ylabel('Number of Patients')
        ax2.set_title('Distribution of Length of Stay')
        st.pyplot(fig2)
        plt.close()
    
    # Row 2: Diagnosis Analysis and Risk Factors
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        st.markdown("### Top 10 Most Common Diagnoses")
        # Get top 10 most common diagnoses
        top_diagnoses = sorted_diagnoses[:10]
        if top_diagnoses:
            diag_names, diag_counts = zip(*top_diagnoses)
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            ax3.barh(range(len(diag_names)), diag_counts, color='#70ad47')
            ax3.set_yticks(range(len(diag_names)))
            ax3.set_yticklabels(diag_names)
            ax3.set_xlabel('Number of Patients')
            ax3.set_title('Top 10 Most Common Diagnoses')
            ax3.invert_yaxis()
            st.pyplot(fig3)
            plt.close()
        else:
            st.info("No diagnosis data available")
    
    with chart_col4:
        st.markdown("### Readmission Rate by Number of Diagnoses")
        # Group by number of diagnoses and calculate readmission rate
        if 'number_diagnoses' in df.columns:
            diagnosis_groups = df.groupby('number_diagnoses')['readmitted_binary'].agg(['count', 'sum']).reset_index()
            diagnosis_groups['readmission_rate'] = (diagnosis_groups['sum'] / diagnosis_groups['count']) * 100
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.plot(diagnosis_groups['number_diagnoses'], diagnosis_groups['readmission_rate'], marker='o', linewidth=2, markersize=8)
            ax4.set_xlabel('Number of Diagnoses')
            ax4.set_ylabel('Readmission Rate (%)')
            ax4.set_title('Readmission Rate by Number of Diagnoses')
            ax4.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig4)
            plt.close()
        else:
            st.info("Number of diagnoses data not available")
    
    st.markdown("---")
    
    # Additional insights
    st.subheader("💡 Key Insights")
    
    # Calculate some additional metrics
    avg_medications_high_risk = df[df['readmitted_binary'] == 1]['num_medications'].mean() if 1 in df['readmitted_binary'].values else 0
    avg_medications_low_risk = df[df['readmitted_binary'] == 0]['num_medications'].mean() if 0 in df['readmitted_binary'].values else 0
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown(f"""
        ##### 🏥 Patient Volume
        - Total patients analyzed: **{total_patients:,}**
        - Readmitted patients: **{readmitted_patients:,}**
        - Average daily admissions: **{total_patients/365:.1f}**
        """)
    
    with col_insight2:
        st.markdown(f"""
        ##### 💊 Medication Insights
        - Avg. medications (High-risk): **{avg_medications_high_risk:.1f}**
        - Avg. medications (Low-risk): **{avg_medications_low_risk:.1f}**
        - Difference: **{abs(avg_medications_high_risk - avg_medications_low_risk):.1f}** meds
        """)
    
    st.markdown("---")
    
    # Data table
    st.subheader("📋 Raw Analytics Data Preview")
    st.dataframe(df[['race', 'gender', 'age', 'time_in_hospital', 'num_medications', 'number_diagnoses', 'readmitted', 'readmitted_binary']].head(10), use_container_width=True)
    
else:
    st.info("Please load the analytics data using the sidebar controls.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Hospital Analytics Dashboard</strong> | 
    Management Tool for Readmission Analytics</p>
    <p><small>Last updated: {}</small></p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)