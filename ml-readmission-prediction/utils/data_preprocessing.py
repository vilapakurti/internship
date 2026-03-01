"""
Data preprocessing module for hospital readmission prediction
Handles data cleaning, feature engineering, and train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        Load the diabetes dataset
        Expected columns based on the dataset description
        """
        try:
            # Load the dataset
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            # Create sample data if file doesn't exist
            return self.create_sample_data()
    
    def create_sample_data(self):
        """
        Create sample data for demonstration purposes
        """
        print("Creating sample data for demonstration...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data based on typical hospital data
        data = {
            'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', '?'], n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Unknown/Invalid'], n_samples),
            'age': np.random.choice(['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], n_samples),
            'admission_type_id': np.random.randint(1, 9, n_samples),
            'discharge_disposition_id': np.random.randint(1, 30, n_samples),
            'admission_source_id': np.random.randint(1, 20, n_samples),
            'time_in_hospital': np.random.randint(1, 15, n_samples),
            'num_lab_procedures': np.random.randint(1, 50, n_samples),
            'num_procedures': np.random.randint(0, 6, n_samples),
            'num_medications': np.random.randint(1, 25, n_samples),
            'number_outpatient': np.random.randint(0, 10, n_samples),
            'number_emergency': np.random.randint(0, 10, n_samples),
            'number_inpatient': np.random.randint(0, 10, n_samples),
            'diag_1': np.random.choice(['250', '428', '414', '403', '276', '427', '584', '250.01', '?'], n_samples),
            'diag_2': np.random.choice(['250', '428', '414', '403', '276', '427', '584', '250.01', '?'], n_samples),
            'diag_3': np.random.choice(['250', '428', '414', '403', '276', '427', '584', '250.01', '?'], n_samples),
            'number_diagnoses': np.random.randint(1, 17, n_samples),
            'metformin': np.random.choice(['Up', 'Down', 'Steady', 'No'], n_samples),
            'insulin': np.random.choice(['Up', 'Down', 'Steady', 'No'], n_samples),
            'prior_admissions': np.random.randint(0, 10, n_samples),
            'insurance_type': np.random.choice(['Private', 'Medicaid', 'Medicare', 'Self-Pay', 'Government', 'Other'], n_samples),
            'chf': np.random.choice([True, False], n_samples),
            'copd': np.random.choice([True, False], n_samples),
            'diabetes': np.random.choice([True, False], n_samples),
            'bp': np.random.choice([True, False], n_samples),
            'ckd': np.random.choice([True, False], n_samples),
            'other_comorb': np.random.choice([True, False], n_samples),
            
            # NEW FEATURES: Enhanced clinical indicators
            'hba1c_result': np.random.choice(['None', 'Norm', 'Abnorm', '>30'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),  # HbA1c result (critical for diabetes)
            'blood_pressure_systolic': np.random.normal(130, 20, n_samples),  # Systolic blood pressure
            'blood_pressure_diastolic': np.random.normal(80, 12, n_samples),  # Diastolic blood pressure
            'bmi': np.random.normal(28, 6, n_samples),  # Body Mass Index
            'creatinine_level': np.random.gamma(2, 1, n_samples),  # Creatinine level (kidney function)
            'length_of_previous_stay': np.random.exponential(5, n_samples),  # Length of previous hospital stay
            'discharged_to_home': np.random.choice([True, False], n_samples),  # Discharged to home vs skilled nursing facility
            'social_risk_factors': np.random.randint(0, 5, n_samples),  # Number of social risk factors (0-4)
            
            # ADVANCED CLINICAL FEATURES based on your suggestions
            'prior_icu_admissions': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.07, 0.03]),  # Prior ICU admissions
            'prior_ed_visits_6m': np.random.poisson(1.2, n_samples),  # Prior ED visits in last 6 months
            'polypharmacy_flag': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),  # Polypharmacy flag (≥5 or ≥10 meds)
            'high_risk_meds': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),  # High-risk medications (e.g., anticoagulants, opioids)
            'med_changes_during_stay': np.random.choice([True, False], n_samples, p=[0.35, 0.65]),  # Medication changes during stay
            'med_adherence_history': np.random.uniform(0, 1, n_samples),  # Medication adherence history (0-1 scale)
            'new_med_started': np.random.choice([True, False], n_samples, p=[0.5, 0.5]),  # New medication started during admission
            'icu_stay_during_admission': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),  # ICU stay during admission
            'mechanical_ventilation': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),  # Mechanical ventilation use
            'sepsis_diagnosis': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),  # Sepsis diagnosis
            'charlson_comorbidity_index': np.random.randint(0, 11, n_samples),  # Charlson Comorbidity Index (composite score)
            'elixhauser_comorbidity_score': np.random.randint(0, 16, n_samples),  # Elixhauser Comorbidity Score
            'abnormal_labs_flag': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),  # Abnormal lab flags
            'vital_instability_score': np.random.uniform(0, 10, n_samples),  # Vital instability score (derived feature)
            'creatinine_change': np.random.normal(0, 0.5, n_samples),  # Change in creatinine (acute kidney injury flag),
            
            # BEHAVIORAL & ENGAGEMENT SIGNALS
            'patient_portal_usage': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2]),  # Patient portal usage (0=never, 1=rarely, 2=sometimes, 3=often)
            'prior_no_show_rate': np.random.beta(1, 5, n_samples),  # Prior no-show rate (0-1 scale)
            'refused_medication': np.random.choice([True, False], n_samples, p=[0.15, 0.85]),  # Refused medication
            'documented_non_compliance': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),  # Documented non-compliance
            'substance_use_disorder': np.random.choice([True, False], n_samples, p=[0.12, 0.88]),  # Substance use disorder history
            'depression_diagnosis': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),  # Depression diagnosis
            'anxiety_diagnosis': np.random.choice([True, False], n_samples, p=[0.18, 0.82]),  # Anxiety diagnosis
            
            # DIAGNOSIS REPRESENTATION FEATURES
            'ccs_diagnosis_category': np.random.randint(0, 25, n_samples),  # CCS diagnosis category
            'chronic_condition_flag': np.random.choice([True, False], n_samples, p=[0.65, 0.35]),  # Chronic condition flag
            'high_risk_diagnosis': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),  # High-risk diagnosis flag (CHF, COPD, pneumonia)
            'count_chronic_conditions': np.random.poisson(2, n_samples),  # Count of chronic conditions
            'drg_code': np.random.randint(1, 1000, n_samples),  # DRG code (Diagnosis Related Group)
            'principal_procedure_type': np.random.randint(0, 50, n_samples),  # Principal procedure type
            
            # DYNAMIC / ENGINEERED FEATURES
            'worsening_comorbidity_indicator': np.random.choice([True, False], n_samples, p=[0.25, 0.75]),  # Worsening comorbidity indicator
            'admission_risk_score_percentile': np.random.uniform(0, 100, n_samples),  # Admission risk score percentile
            'age_chf_interaction': np.random.uniform(0, 100, n_samples),  # Interaction term (Age × CHF)
            'age_copd_interaction': np.random.uniform(0, 100, n_samples),  # Interaction term (Age × COPD),
        }
        
        # Adjust blood pressure values to realistic ranges
        data['blood_pressure_systolic'] = np.clip(data['blood_pressure_systolic'], 90, 200)
        data['blood_pressure_diastolic'] = np.clip(data['blood_pressure_diastolic'], 60, 120)
        
        # Adjust BMI to realistic range
        data['bmi'] = np.clip(data['bmi'], 15, 50)
        
        # Adjust creatinine to realistic range
        data['creatinine_level'] = np.clip(data['creatinine_level'], 0.5, 5.0)
        
        # Adjust length of previous stay to realistic range
        data['length_of_previous_stay'] = np.clip(data['length_of_previous_stay'], 0, 30)
        
        # Create target variable based on comprehensive risk factors
        # Higher chance of readmission if multiple risk factors are present
        risk_factors = (
            (data['time_in_hospital'] > 7).astype(int) * 0.8 +  # Extended hospital stay
            (data['num_medications'] > 15).astype(int) * 0.6 +  # Polypharmacy
            (data['number_diagnoses'] > 8).astype(int) * 0.7 +  # Multiple comorbidities
            (data['number_inpatient'] > 2).astype(int) * 0.5 +  # Frequent hospitalizations
            (data['prior_admissions'] > 2).astype(int) * 0.9 +  # Prior admissions
            (data['chf']).astype(int) * 1.0 +  # Congestive heart failure
            (data['copd']).astype(int) * 0.7 +  # COPD
            (data['diabetes']).astype(int) * 0.6 +  # Diabetes
            (data['bp']).astype(int) * 0.4 +  # Hypertension
            (data['ckd']).astype(int) * 0.8 +  # Chronic kidney disease
            # NEW FEATURES CONTRIBUTIONS:
            (np.isin(data['hba1c_result'], ['Abnorm', '>30'])).astype(int) * 0.8 +  # Abnormal HbA1c
            ((data['blood_pressure_systolic'] > 160) | (data['blood_pressure_diastolic'] > 100)).astype(int) * 0.5 +  # High BP
            (data['bmi'] > 35).astype(int) * 0.4 +  # High BMI (obesity)
            (data['creatinine_level'] > 2.0).astype(int) * 0.7 +  # High creatinine (kidney issues)
            (data['length_of_previous_stay'] > 10).astype(int) * 0.6 +  # Long previous stay
            (~data['discharged_to_home']).astype(int) * 0.5 +  # Not discharged to home
            (data['social_risk_factors'] > 2).astype(int) * 0.6 +  # High social risk
            # ADVANCED CLINICAL FEATURES CONTRIBUTIONS:
            (data['prior_icu_admissions'] > 0).astype(int) * 1.2 +  # Prior ICU admissions
            (data['prior_ed_visits_6m'] > 2).astype(int) * 0.6 +  # Prior ED visits in last 6 months
            (data['polypharmacy_flag']).astype(int) * 0.7 +  # Polypharmacy flag
            (data['high_risk_meds']).astype(int) * 0.8 +  # High-risk medications
            (data['med_changes_during_stay']).astype(int) * 0.5 +  # Medication changes during stay
            ((1 - data['med_adherence_history']) > 0.5).astype(int) * 0.9 +  # Poor medication adherence
            (data['new_med_started']).astype(int) * 0.4 +  # New medication started during admission
            (data['icu_stay_during_admission']).astype(int) * 1.0 +  # ICU stay during admission
            (data['mechanical_ventilation']).astype(int) * 1.1 +  # Mechanical ventilation use
            (data['sepsis_diagnosis']).astype(int) * 1.3 +  # Sepsis diagnosis
            (data['charlson_comorbidity_index'] > 5).astype(int) * 0.8 +  # High Charlson Comorbidity Index
            (data['elixhauser_comorbidity_score'] > 8).astype(int) * 0.7 +  # High Elixhauser Comorbidity Score
            (data['abnormal_labs_flag']).astype(int) * 0.6 +  # Abnormal lab flags
            (data['vital_instability_score'] > 6).astype(int) * 0.9 +  # High vital instability score
            (abs(data['creatinine_change']) > 0.5).astype(int) * 0.7 +  # Significant change in creatinine
            # BEHAVIORAL & ENGAGEMENT SIGNALS CONTRIBUTIONS:
            (data['patient_portal_usage'] < 2).astype(int) * 0.4 +  # Low patient portal usage
            (data['prior_no_show_rate'] > 0.3).astype(int) * 0.8 +  # High no-show rate
            (data['refused_medication']).astype(int) * 1.0 +  # Refused medication
            (data['documented_non_compliance']).astype(int) * 1.1 +  # Documented non-compliance
            (data['substance_use_disorder']).astype(int) * 0.9 +  # Substance use disorder
            (data['depression_diagnosis']).astype(int) * 0.7 +  # Depression diagnosis
            (data['anxiety_diagnosis']).astype(int) * 0.5 +  # Anxiety diagnosis
            # DIAGNOSIS REPRESENTATION FEATURES CONTRIBUTIONS:
            (data['chronic_condition_flag']).astype(int) * 0.6 +  # Chronic condition flag
            (data['high_risk_diagnosis']).astype(int) * 1.0 +  # High-risk diagnosis
            (data['count_chronic_conditions'] > 4).astype(int) * 0.7 +  # Count of chronic conditions
            # DYNAMIC / ENGINEERED FEATURES CONTRIBUTIONS:
            (data['worsening_comorbidity_indicator']).astype(int) * 0.9 +  # Worsening comorbidity
            (data['admission_risk_score_percentile'] > 75).astype(int) * 0.8 +  # High risk score percentile
            (data['age_chf_interaction'] > 50).astype(int) * 0.7 +  # Age-CHF interaction
            (data['age_copd_interaction'] > 50).astype(int) * 0.6  # Age-COPD interaction
        )
        
        # Adjust probabilities based on insurance type (Medicaid/Medicare may have higher readmission rates)
        insurance_factor = np.array([1.2 if ins in ['Medicaid', 'Medicare'] else 0.9 for ins in data['insurance_type']])
        
        # Create readmission probability based on risk factors and insurance
        base_prob = 0.05 + (risk_factors / 15.0)  # Normalize risk factors
        adjusted_prob = base_prob * insurance_factor
        adjusted_prob = np.clip(adjusted_prob, 0.05, 0.95)  # Keep probabilities reasonable
        
        readmitted = [np.random.choice([0, 1], p=[1-p, p]) for p in adjusted_prob]
        
        data['readmitted'] = ['<30' if r == 1 else '>30' for r in readmitted]
        
        df = pd.DataFrame(data)
        print(f"Sample dataset created with shape: {df.shape}")
        print(f"New features added: hba1c_result, blood_pressure_systolic, blood_pressure_diastolic, bmi, creatinine_level, length_of_previous_stay, discharged_to_home, social_risk_factors, prior_icu_admissions, prior_ed_visits_6m, polypharmacy_flag, high_risk_meds, med_changes_during_stay, med_adherence_history, new_med_started, icu_stay_during_admission, mechanical_ventilation, sepsis_diagnosis, charlson_comorbidity_index, elixhauser_comorbidity_score, abnormal_labs_flag, vital_instability_score, creatinine_change, patient_portal_usage, prior_no_show_rate, refused_medication, documented_non_compliance, substance_use_disorder, depression_diagnosis, anxiety_diagnosis, ccs_diagnosis_category, chronic_condition_flag, high_risk_diagnosis, count_chronic_conditions, drg_code, principal_procedure_type, worsening_comorbidity_indicator, admission_risk_score_percentile, age_chf_interaction, age_copd_interaction")
        return df
    
    def preprocess_data(self, df):
        """
        Perform data preprocessing steps:
        - Handle missing values
        - Convert categorical variables
        - Feature engineering
        - Convert target to binary
        """
        print("Starting data preprocessing...")
        
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Convert target variable to binary
        # '<30' -> 1 (high risk), '>30'/'NO' -> 0 (low risk)
        df_processed['readmitted_binary'] = df_processed['readmitted'].apply(
            lambda x: 1 if x == '<30' else 0
        )
        
        # Drop original readmitted column
        df_processed = df_processed.drop('readmitted', axis=1)
        
        # Handle boolean columns by converting them to integers (True->1, False->0)
        boolean_cols = df_processed.select_dtypes(include=['bool']).columns.tolist()
        for col in boolean_cols:
            df_processed[col] = df_processed[col].astype(int)
        
        # Identify categorical columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables using Label Encoding
        for col in categorical_cols:
            if col != 'readmitted_binary':  # Skip target variable
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Encoded {len(categorical_cols)} categorical columns")
        print(f"Converted {len(boolean_cols)} boolean columns to integer")
        
        # Separate features and target
        X = df_processed.drop('readmitted_binary', axis=1)
        y = df_processed['readmitted_binary']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        print(f"Splitting data with test size: {test_size}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        """
        print("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to maintain column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load data (will create sample data if file doesn't exist)
    df = preprocessor.load_data("data/diabetes.csv")  # This will trigger sample data creation
    
    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    print("\nTarget distribution:")
    print(df['readmitted'].value_counts() if 'readmitted' in df.columns else "Column not found")
    
    # Preprocess the data
    X, y = preprocessor.preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Scale the features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print("\nPreprocessing completed successfully!")
    print(f"Final training set shape: {X_train_scaled.shape}")
    print(f"Final testing set shape: {X_test_scaled.shape}")