"""
Main script for training transit detection models.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add the project root directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.data.data_loader import list_fits_files, load_lightcurve, clean_lightcurve, normalize_flux
from src.features.feature_extraction import identify_potential_transits
from src.features.feature_builder import build_dataset_from_lightcurves, create_dataframe, balance_dataset
from src.models.random_forest import create_rf_pipeline, evaluate_rf_model
from src.visualization.visualize import plot_feature_importances, plot_confusion_matrix, plot_precision_recall_curve

def main():
    """
    Main function to train transit detection models.
    """
    # Parameters
    data_dir = os.path.join(project_root, "data", "raw")
    threshold_percentile = 5
    test_size = 0.3
    random_state = 42
    
    print("Loading FITS files...")
    fits_files = list_fits_files(data_dir)
    print(f"Found {len(fits_files)} FITS files")
    
    # Process files and extract features
    all_features, all_labels, all_sources = [], [], []
    
    for path in fits_files:
        # Load and clean data
        time, flux, quality = load_lightcurve(path)
        time_clean, flux_clean = clean_lightcurve(time, flux, quality)
        
        if time_clean is None or len(time_clean) == 0:
            continue
            
        # Normalize flux
        flux_norm = normalize_flux(flux_clean)
        
        # Identify potential transit points
        _, dip_mask, dip_indices = identify_potential_transits(flux_norm, threshold_percentile)
        
        # Extract features
        features, labels = build_dataset_from_lightcurves(time_clean, flux_norm, dip_indices, dip_mask)
        
        # Add to collection
        all_features.extend(features)
        all_labels.extend(labels)
        all_sources.extend([path] * len(features))
    
    # Create DataFrame
    print("Creating dataset...")
    df = create_dataframe(all_features, all_labels, all_sources)
    
    # Print label distribution
    print("Label distribution:")
    print(df.label.value_counts())
    
    # Split by source file to prevent data leakage
    print("Splitting data...")
    unique_files = df['source_file'].unique()
    train_files, test_files = train_test_split(unique_files, test_size=test_size, random_state=random_state)
    
    train_mask = df['source_file'].isin(train_files)
    X_train = df.loc[train_mask, df.columns[:-2]]  # Exclude 'label' and 'source_file'
    y_train = df.loc[train_mask, 'label']
    X_test = df.loc[~train_mask, df.columns[:-2]]
    y_test = df.loc[~train_mask, 'label']
    
    # Balance training set if needed
    if y_train.value_counts().min() < 0.5 * y_train.value_counts().max():
        print("Balancing training set...")
        X_train, y_train = balance_dataset(X_train, y_train)
    
    # Create and evaluate Random Forest model
    print("Training Random Forest model...")
    rf_pipeline = create_rf_pipeline()
    rf_results = evaluate_rf_model(rf_pipeline, X_train, y_train, X_test, y_test)
    
    # Print results
    print(f"\nRandom Forest cross-validation accuracy: {rf_results['cv_score']['mean_accuracy']:.3f} "
          f"± {rf_results['cv_score']['std_accuracy']:.3f}")
    print("\nClassification Report:")
    for label, metrics in rf_results['classification_report'].items():
        if label in ['0', '1']:
            print(f"Class {label}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Plot results
    feature_names = X_train.columns.tolist()
    plot_feature_importances(rf_results['feature_importances'], feature_names)
    plot_confusion_matrix(rf_results['confusion_matrix'])
    
    # Optional: train a neural network if TensorFlow is available
    try:
        from src.models.neural_network import create_neural_network_model, train_neural_network, evaluate_neural_network
        
        print("\nTraining Neural Network model...")
        nn_model = create_neural_network_model((X_train.shape[1],))
        nn_model, history, scaler = train_neural_network(nn_model, X_train, y_train)
        nn_results = evaluate_neural_network(nn_model, X_test, y_test, scaler)
        
        print("\nNeural Network Classification Report:")
        for label, metrics in nn_results['classification_report'].items():
            if label in ['0', '1']:
                print(f"Class {label}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        # Plot NN results
        from src.visualization.visualize import plot_neural_network_history
        plot_neural_network_history(history)
        plot_confusion_matrix(nn_results['confusion_matrix'], "Neural Network Confusion Matrix")
        plot_precision_recall_curve(y_test, nn_results['test_probabilities'])
        
    except ImportError:
        print("\nSkipping Neural Network model (TensorFlow not available)")
    
    print("\nDone!")

if __name__ == "__main__":
    main()