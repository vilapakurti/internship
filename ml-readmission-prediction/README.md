# Hospital Readmission Prediction Project

This project implements a machine learning model to predict 30-day hospital readmissions using the Diabetes 130-US Hospitals dataset. The goal is to identify patients at high risk of readmission so that preventive care can be provided before discharge.

## Project Structure

```
ml-readmission-prediction/
├── data/                    # Data files
├── models/                  # Model training scripts
│   └── model_training.py    # Contains model training and evaluation functions
├── utils/                   # Utility scripts
│   └── data_preprocessing.py # Data preprocessing functions
├── results/                 # Output results and visualizations
├── main.py                  # Main execution script
├── app.py                   # Streamlit web application
├── web_app.py               # Alternative web application
├── hospital_readmission_prediction.ipynb  # Jupyter notebook with complete workflow
├── requirements.txt         # Project dependencies
├── README.md               # This file
├── run_project.bat         # Windows batch script to run the project
└── run_web_app.bat         # Windows batch script to run the web application
```

## Features

- Data preprocessing and cleaning
- Feature engineering
- Train/test split (80/20)
- Implementation of two classification algorithms:
  - Logistic Regression
  - Random Forest Classifier
- Comprehensive model evaluation using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Visualization of results
- Feature importance analysis
- Model persistence (saving/loading)
- Interactive web application with Streamlit
- Real-time patient risk assessment
- User-friendly interface for healthcare professionals

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the main script

```bash
python main.py
```

### Option 2: Execute the Jupyter notebook

```bash
jupyter notebook hospital_readmission_prediction.ipynb
```

### Option 3: Launch the Web Application

```bash
streamlit run app.py
```

Or use the batch file for Windows:

```bash
run_web_app.bat
```

The web application provides an interactive interface where healthcare professionals can input patient data and receive immediate risk assessments.

## Algorithms Used

### Logistic Regression
- Binary classification algorithm
- Simple and efficient
- Works well for linear relationships

### Random Forest Classifier
- Ensemble learning method using multiple decision trees
- Reduces overfitting
- Handles complex datasets effectively

## Model Evaluation Metrics

- **Accuracy**: (Correct Predictions) / (Total Predictions)
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Shows True Positives, True Negatives, False Positives, False Negatives

## Dataset Description

The model uses the Diabetes 130-US Hospitals Dataset (1999-2008) with features including:
- Patient demographics (age, gender)
- Admission type
- Diagnosis codes
- Lab procedures
- Medications
- Readmission status

Target Variable: readmitted
- `<30` → 1 (High Risk)
- `>30` / `NO` → 0 (Low Risk)

## Expected Results

Based on the project report:
- Logistic Regression Accuracy: ~70%
- Random Forest Accuracy: ~75%
- Random Forest typically performs better than Logistic Regression

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Web Application Features

The Streamlit web application includes:

- Interactive patient data input forms
- Real-time risk prediction
- Model selection (Logistic Regression or Random Forest)
- Probability scores for both risk categories
- Risk factor analysis
- Patient data summary
- Model performance metrics display
- Responsive design for various screen sizes

## Project Objectives Achieved

✓ Analyze hospital patient data
✓ Perform data preprocessing and cleaning
✓ Develop Machine Learning classification models
✓ Predict 30-day readmission risk
✓ Evaluate model performance using standard metrics
✓ Deploy model with user-friendly web interface

## Future Scope

- Deploy the system on cloud platforms
- Integration with hospital management systems
- Real-time prediction and monitoring
- Use of advanced deep learning models
- Improved feature engineering for higher accuracy

## References

1. Kaggle – Diabetes 130-US Hospitals Dataset
2. Scikit-learn Documentation
3. Research papers on Hospital Readmission Prediction