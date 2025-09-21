"""
Main script for training stellar classification models.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add the project root directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.features.feature_extraction import apply_feature_selection, analyze_feature_importance
from src.models.random_forest import create_rf_pipeline, evaluate_rf_model
from src.models.svm import create_svm_pipeline, evaluate_svm_model
from src.evaluation.model_evaluation import evaluate_multiple_models
from src.visualization.visualize import (
    plot_feature_importances, plot_confusion_matrix, 
    plot_class_distribution, plot_feature_correlation
)

def main():
    """
    Main function to train stellar classification models.
    """
    # Parameters
    data_file = os.path.join(project_root, "data", "processed", "stellar_features.csv")
    test_size = 0.3
    random_state = 42
    
    # Load data
    print("Loading dataset...")
    try:
        df = pd.read_csv(data_file)
        print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    except FileNotFoundError:
        print(f"Dataset file not found: {data_file}")
        print("Please run make_dataset.py first to prepare the data.")
        return
    
    # Split features and target
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Plot class distribution
    plot_class_distribution(y)
    
    # Analyze feature correlations
    plot_feature_correlation(X)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(X, y)
    print("\nTop 10 features by importance:")
    print(importance_df.head(10))
    
    # Apply feature selection
    print("\nApplying feature selection...")
    X_selected, selector = apply_feature_selection(X, y, n_features=15, method='selectk')
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train and evaluate Random Forest model
    print("\nTraining Random Forest model...")
    rf_pipeline = create_rf_pipeline()
    rf_results = evaluate_rf_model(rf_pipeline, X_train, y_train, X_test, y_test)
    
    # Train and evaluate SVM model
    print("\nTraining SVM model...")
    svm_pipeline = create_svm_pipeline()
    svm_results = evaluate_svm_model(svm_pipeline, X_train, y_train, X_test, y_test)
    
    # Compare models
    models_with_predictions = [
        ("Random Forest", rf_results['test_predictions'], rf_results['test_probabilities']),
        ("SVM", svm_results['test_predictions'], svm_results['test_probabilities'])
    ]
    
    print("\nModel Comparison:")
    comparison_df = evaluate_multiple_models(models_with_predictions, y_test)
    print(comparison_df)
    
    # Plot confusion matrices
    plot_confusion_matrix(rf_results['confusion_matrix'], "Random Forest Confusion Matrix")
    plot_confusion_matrix(svm_results['confusion_matrix'], "SVM Confusion Matrix")
    
    # Plot feature importances for Random Forest
    if hasattr(rf_pipeline.named_steps['clf'], 'feature_importances_'):
        feature_names = X_selected.columns
        plot_feature_importances(
            rf_pipeline.named_steps['clf'].feature_importances_, 
            feature_names
        )
    
    # Optional: train a neural network if TensorFlow is available
    try:
        from src.models.neural_network import create_nn_model, train_nn_model, evaluate_nn_model
        
        print("\nTraining Neural Network model...")
        nn_model = create_nn_model(input_shape=(X_train.shape[1],), num_classes=len(np.unique(y)))
        nn_model, history = train_nn_model(nn_model, X_train, y_train)
        nn_results = evaluate_nn_model(nn_model, X_test, y_test)
        
        # Add Neural Network to model comparison
        models_with_predictions.append(
            ("Neural Network", nn_results['test_predictions'], nn_results['test_probabilities'])
        )
        
        # Update comparison
        print("\nUpdated Model Comparison (with Neural Network):")
        comparison_df = evaluate_multiple_models(models_with_predictions, y_test)
        print(comparison_df)
        
        # Plot Neural Network results
        from src.visualization.visualize import plot_nn_history
        plot_nn_history(history)
        plot_confusion_matrix(nn_results['confusion_matrix'], "Neural Network Confusion Matrix")
        
    except ImportError:
        print("\nSkipping Neural Network model (TensorFlow not available)")
    
    print("\nDone!")

if __name__ == "__main__":
    main()