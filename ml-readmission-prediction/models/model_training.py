"""
Model training module for hospital readmission prediction
Implements Logistic Regression and Random Forest classifiers
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
import shap


class ReadmissionPredictor:
    def __init__(self):
        self.logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.models = {
            'Logistic Regression': self.logistic_model,
            'Random Forest': self.rf_model
        }
        self.trained_models = {}
        self.calibrated_models = {}
        self.results = {}
        self.X_train = None  # Store training data for SHAP explanations
        self.y_train = None  # Store training labels for calibration
    
    def train_models(self, X_train, y_train):
        """
        Train all models using time-aware cross-validation and probability calibration
        """
        print("Training models with time-aware cross-validation and calibration...")
        
        # Store the training data for SHAP explanations and calibration
        self.X_train = X_train
        self.y_train = y_train
        
        # Use TimeSeriesSplit for time-aware cross-validation to avoid data leakage
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train the base model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Calibrate the model probabilities using cross-validation
            calibrated_model = CalibratedClassifierCV(model, cv=tscv, method='isotonic')
            calibrated_model.fit(X_train, y_train)
            self.calibrated_models[name] = calibrated_model
            
            print(f"{name} training and calibration completed.")
        
        print("All models trained and calibrated successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate both models using multiple metrics
        """
        print("Evaluating models...")
        
        for name, model in self.trained_models.items():
            print(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print()
        
        return self.results
    
    def plot_confusion_matrices(self, save_path=None):
        """
        Plot confusion matrices for both models
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Confusion Matrices for Readmission Prediction Models')
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\n(Accuracy: {result["accuracy"]:.3f})')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, save_path=None):
        """
        Plot comparison of model metrics
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        # Extract metric values
        metric_values = {metric: [self.results[model][metric] for model in model_names] 
                         for metric in metrics}
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, metric in enumerate(metrics):
            values = metric_values[metric]
            ax.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, metric in enumerate(metrics):
            for j, value in enumerate(metric_values[metric]):
                ax.text(j + i*width, value + 0.01, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_models(self, save_dir='models'):
        """
        Save trained models to disk
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.trained_models.items():
            model_path = os.path.join(save_dir, f'{name.lower().replace(" ", "_")}_model.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {name} model to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from disk
        """
        return joblib.load(model_path)
    
    def predict_readmission_risk(self, model_name, patient_data):
        """
        Predict readmission risk for a single patient
        """
        if model_name not in self.calibrated_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.calibrated_models.keys())}")
        
        # Use calibrated model for better probability estimates
        model = self.calibrated_models[model_name]
        
        # Handle feature names issue by converting to numpy array if needed
        if hasattr(patient_data, 'values'):
            # If it's a DataFrame/pandas Series, extract the values
            patient_array = patient_data.values
        else:
            # If it's already a numpy array or list
            patient_array = patient_data
            
        # Suppress feature names warning by using numpy array
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            prediction = model.predict(patient_array.reshape(1, -1))[0]
            probability = model.predict_proba(patient_array.reshape(1, -1))[0]
        
        risk_level = "High Risk (<30 days)" if prediction == 1 else "Low Risk (>30 days or No)"
        
        return {
            'prediction': prediction,
            'risk_level': risk_level,
            'probability': {
                'low_risk_prob': probability[0],
                'high_risk_prob': probability[1]
            }
        }

    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance for a trained model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        model = self.trained_models[model_name]
        
        if model_name == 'Random Forest':
            importances = model.feature_importances_
        elif model_name == 'Logistic Regression':
            importances = abs(model.coef_[0])
        else:
            # For calibrated models, use the base estimator
            if hasattr(model, 'base_estimator_'):
                base_model = model.base_estimator_
                if hasattr(base_model, 'feature_importances_'):
                    importances = base_model.feature_importances_
                elif hasattr(base_model, 'coef_'):
                    importances = abs(base_model.coef_[0])
                else:
                    raise ValueError(f"Feature importance not available for model: {model_name}")
            else:
                raise ValueError(f"Feature importance not available for model: {model_name}")
        
        # Create a dataframe with feature names and their importance scores
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        return importance_df
    
    def get_shap_explanation(self, model_name, patient_data):
        """
        Get SHAP explanation for a single patient prediction
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
        
        if self.X_train is None:
            raise ValueError("Training data not available. Please train the model first.")
        
        model = self.trained_models[model_name]
        
        # Handle feature names issue by converting to numpy array if needed
        if hasattr(patient_data, 'values'):
            # If it's a DataFrame/pandas Series, extract the values
            patient_array = patient_data.values
        else:
            # If it's already a numpy array or list
            patient_array = patient_data
        
        # Create SHAP explainer based on model type
        if model_name == 'Random Forest':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(model, self.X_train)
        else:
            # For other models, use the permutation explainer
            explainer = shap.PermutationExplainer(model.predict_proba, self.X_train)
        
        # Get SHAP values for the patient
        shap_values = explainer.shap_values(patient_array.reshape(1, -1))
        
        # For binary classification, we usually want the SHAP values for the positive class
        if len(shap_values) == 2:  # Binary classification
            shap_values = shap_values[:, :, 1]  # Take positive class (index 1)
        
        return shap_values, explainer.expected_value
    
    def get_top_risk_factors(self, model_name, feature_names, patient_data, top_n=5):
        """
        Get top N risk factors contributing to the prediction
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.trained_models.keys())}")
            
        model = self.trained_models[model_name]
            
        # For tree-based models, we can use feature importance directly
        if model_name == 'Random Forest':
            importances = model.feature_importances_
                    
            # Get the patient's feature values
            if hasattr(patient_data, 'values'):
                patient_values = patient_data.values
            else:
                patient_values = patient_data
                        
            # Create a dataframe with feature names, importance, and patient values
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances,
                'value': patient_values
            })
                    
            # Sort by importance and get top N
            top_factors = importance_df.nlargest(top_n, 'importance')
                    
            return top_factors
            
        # For Logistic Regression, we can use coefficients and patient values
        elif model_name == 'Logistic Regression':
            coefficients = model.coef_[0]
                
            # Get the patient's feature values
            if hasattr(patient_data, 'values'):
                patient_values = patient_data.values
            else:
                patient_values = patient_data
                
            # Calculate contribution of each feature to the prediction
            contributions = abs(coefficients * patient_values)
                
            # Create a dataframe with feature names, contributions, coefficients, and patient values
            contribution_df = pd.DataFrame({
                'feature': feature_names,
                'contribution': contributions,
                'coefficient': coefficients,
                'value': patient_values
            })
                
            # Sort by contribution and get top N
            top_factors = contribution_df.nlargest(top_n, 'contribution')
                
            return top_factors
            
        else:
            raise ValueError(f"Top risk factors not available for model: {model_name}")

    def train_diagnosis_specific_models(self, X_train, y_train, diag_column='diag_1'):
        """
        Train separate models for high-volume diagnoses (CHF, COPD, Diabetes)
        """
        print("Training diagnosis-specific models...")
        
        # Get unique diagnosis values
        unique_diagnoses = X_train[diag_column].unique()
        
        # Define high-volume diagnoses (based on common readmission causes)
        high_volume_diagnoses = ['428', '491', '250']  # Heart failure, COPD, Diabetes
        
        self.diagnosis_models = {}
        
        for diag in high_volume_diagnoses:
            if diag in unique_diagnoses:
                # Filter data for this specific diagnosis
                mask = X_train[diag_column] == diag
                X_diag = X_train[mask]
                y_diag = y_train[mask]
                
                if len(y_diag) > 10:  # Only train if we have enough samples
                    print(f"Training model for diagnosis: {diag} (samples: {len(y_diag)})")
                    
                    # Train a specialized model for this diagnosis
                    diag_model = {
                        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
                    }
                    
                    # Train each model type for this diagnosis
                    diag_model_trained = {}
                    for name, model in diag_model.items():
                        model.fit(X_diag.drop(columns=[diag_column]), y_diag)
                        diag_model_trained[name] = model
                    
                    # Store the trained models for this diagnosis
                    self.diagnosis_models[diag] = diag_model_trained
                else:
                    print(f"Skipping diagnosis {diag} - insufficient samples ({len(y_diag)})")
        
        print(f"Trained specialized models for {len(self.diagnosis_models)} high-volume diagnoses")


def print_results_table(results):
    """
    Print a formatted table of model results
    """
    print("="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*70)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    print("="*70)


if __name__ == "__main__":
    # This would typically be run after preprocessing
    print("Model training module ready.")
    print("To use this module, initialize the ReadmissionPredictor class and call train_models().")