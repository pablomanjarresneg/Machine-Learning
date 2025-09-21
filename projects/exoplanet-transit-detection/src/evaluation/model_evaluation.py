"""
Model evaluation utilities for transit detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

def evaluate_model_predictions(y_true, y_pred, y_proba=None, model_name="Model"):
    """
    Evaluate model predictions and return performance metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_proba (array): Predicted probabilities (optional)
        model_name (str): Name of the model for reporting
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    # If probabilities are provided, compute AUC and average precision
    if y_proba is not None:
        # For neural networks, ensure shape is correct
        if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1 and y_proba.shape[1] == 1:
            y_proba = y_proba.flatten()
            
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        metrics['auc'] = auc(fpr, tpr)
        
        # Precision-recall curve and average precision
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics

def evaluate_multiple_models(models_with_predictions, y_true):
    """
    Evaluate and compare multiple models.
    
    Args:
        models_with_predictions (list): List of tuples (model_name, y_pred, y_proba)
        y_true (array): True labels
        
    Returns:
        DataFrame: DataFrame with evaluation metrics for each model
    """
    all_metrics = []
    
    for model_tuple in models_with_predictions:
        if len(model_tuple) == 3:
            model_name, y_pred, y_proba = model_tuple
        else:
            model_name, y_pred = model_tuple
            y_proba = None
        
        metrics = evaluate_model_predictions(y_true, y_pred, y_proba, model_name)
        all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)

def find_optimal_threshold(y_true, y_proba):
    """
    Find the optimal classification threshold for imbalanced data.
    
    Args:
        y_true (array): True labels
        y_proba (array): Predicted probabilities
        
    Returns:
        tuple: (optimal_threshold, metrics at threshold)
    """
    # Calculate precision and recall at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Get metrics at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    return optimal_threshold, metrics

def plot_all_roc_curves(models_with_predictions, y_true):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_with_predictions (list): List of tuples (model_name, y_pred, y_proba)
        y_true (array): True labels
    """
    plt.figure(figsize=(10, 8))
    
    for model_tuple in models_with_predictions:
        if len(model_tuple) < 3:
            continue
            
        model_name, _, y_proba = model_tuple
        
        # For neural networks, ensure shape is correct
        if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1 and y_proba.shape[1] == 1:
            y_proba = y_proba.flatten()
            
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_all_precision_recall_curves(models_with_predictions, y_true):
    """
    Plot precision-recall curves for multiple models.
    
    Args:
        models_with_predictions (list): List of tuples (model_name, y_pred, y_proba)
        y_true (array): True labels
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate baseline for imbalanced data
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--', alpha=0.8, 
                label=f'Baseline (Class balance = {baseline:.3f})')
    
    for model_tuple in models_with_predictions:
        if len(model_tuple) < 3:
            continue
            
        model_name, _, y_proba = model_tuple
        
        # For neural networks, ensure shape is correct
        if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1 and y_proba.shape[1] == 1:
            y_proba = y_proba.flatten()
            
        # Calculate precision-recall curve and average precision
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()