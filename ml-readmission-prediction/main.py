"""
Main execution script for hospital readmission prediction project
This script orchestrates the entire pipeline from data preprocessing to model evaluation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from utils.data_preprocessing import DataPreprocessor
from models.model_training import ReadmissionPredictor, print_results_table
import pandas as pd
import numpy as np


def main():
    print("="*60)
    print("HOSPITAL READMISSION PREDICTION PROJECT")
    print("="*60)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    predictor = ReadmissionPredictor()
    
    print("\n1. Loading and Preprocessing Data...")
    
    # Load data (creates sample data if file doesn't exist)
    df = preprocessor.load_data("data/diabetes.csv")
    
    # Preprocess the data
    X, y = preprocessor.preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Scale the features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    print("\n2. Training Models...")
    
    # Train both models
    predictor.train_models(X_train_scaled, y_train)
    
    print("\n3. Evaluating Models...")
    
    # Evaluate both models
    results = predictor.evaluate_models(X_test_scaled, y_test)
    
    # Print results table
    print_results_table(results)
    
    print("\n4. Visualizing Results...")
    
    # Create visualizations
    try:
        predictor.plot_confusion_matrices(save_path="results/confusion_matrices.png")
        predictor.plot_model_comparison(save_path="results/model_comparison.png")
        print("Visualizations saved to results/ directory")
    except Exception as e:
        print(f"Could not create visualizations: {e}")
        print("This might be because you're running in a headless environment")
    
    print("\n5. Saving Models...")
    
    # Save the trained models
    predictor.save_models()
    
    print("\n6. Sample Predictions...")
    
    # Demonstrate prediction on a sample patient
    sample_patient = X_test_scaled.iloc[0].values
    model_name = "Random Forest"  # Using the better performing model
    
    try:
        prediction_result = predictor.predict_readmission_risk(model_name, sample_patient)
        print(f"\nSample prediction for {model_name}:")
        print(f"Prediction: {prediction_result['prediction']} ({prediction_result['risk_level']})")
        print(f"Probabilities - Low Risk: {prediction_result['probability']['low_risk_prob']:.3f}, "
              f"High Risk: {prediction_result['probability']['high_risk_prob']:.3f}")
    except Exception as e:
        print(f"Could not make sample prediction: {e}")
    
    print("\n" + "="*60)
    print("PROJECT EXECUTION COMPLETED")
    print("="*60)
    
    # Identify the best performing model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    print(f"\nBest Performing Model: {best_model}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    print("\nNext Steps:")
    print("- Analyze feature importance to understand key factors for readmission")
    print("- Fine-tune hyperparameters for improved performance")
    print("- Validate on additional datasets if available")
    print("- Deploy model for real-time predictions")


def analyze_feature_importance():
    """
    Analyze and display feature importance from the Random Forest model
    """
    print("\nAnalyzing Feature Importance...")
    
    # This would typically be called after training
    # For now, we'll show how it would be done
    print("Feature importance analysis would be performed here.")
    print("With a real dataset, we would extract and visualize the most important features.")


if __name__ == "__main__":
    main()
    
    # Uncomment the line below to also run feature importance analysis
    # analyze_feature_importance()