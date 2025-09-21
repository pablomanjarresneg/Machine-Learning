"""
Common model evaluation metrics for all projects.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def classification_metrics(y_true, y_pred, y_proba=None, average='weighted'):
    """
    Calculate common classification metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        y_proba (array): Predicted probabilities (optional)
        average (str): Averaging method for multi-class metrics
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Add AUC if probabilities are provided (binary classification)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # For 2D probability arrays, use the second column (probability of class 1)
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
    
    return metrics

def regression_metrics(y_true, y_pred):
    """
    Calculate common regression metrics.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def compare_models(models_with_predictions, y_true, task='classification'):
    """
    Compare multiple models using common metrics.
    
    Args:
        models_with_predictions (list): List of tuples (model_name, y_pred, y_proba)
                                        For regression, use (model_name, y_pred)
        y_true (array): True values/labels
        task (str): 'classification' or 'regression'
        
    Returns:
        DataFrame: DataFrame with model comparison
    """
    results = []
    
    for model_tuple in models_with_predictions:
        if task == 'classification':
            if len(model_tuple) == 3:
                model_name, y_pred, y_proba = model_tuple
                metrics = classification_metrics(y_true, y_pred, y_proba)
            else:
                model_name, y_pred = model_tuple
                metrics = classification_metrics(y_true, y_pred)
        else:  # regression
            model_name, y_pred = model_tuple
            metrics = regression_metrics(y_true, y_pred)
        
        metrics['model'] = model_name
        results.append(metrics)
    
    return pd.DataFrame(results)

def cross_validation_summary(cv_results):
    """
    Summarize cross-validation results.
    
    Args:
        cv_results (dict): Cross-validation results from scikit-learn
        
    Returns:
        DataFrame: Summary of results
    """
    # Get all metrics from the results
    metrics = [key for key in cv_results.keys() if key.startswith('test_')]
    
    # Calculate mean and std for each metric
    summary = {}
    for metric in metrics:
        mean_key = f"mean_{metric[5:]}"  # Remove 'test_' prefix
        std_key = f"std_{metric[5:]}"
        summary[mean_key] = np.mean(cv_results[metric])
        summary[std_key] = np.std(cv_results[metric])
    
    return pd.DataFrame([summary])