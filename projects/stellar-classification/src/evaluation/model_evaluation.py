"""
Model evaluation utilities for stellar classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
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
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multi-class classification
    if len(np.unique(y_true)) > 2:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # If probabilities are provided, compute AUC for binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # For neural networks, ensure shape is correct
        if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1 and y_proba.shape[1] == 1:
            y_proba = y_proba.flatten()
        elif hasattr(y_proba, 'shape') and len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
            # If binary probabilities are provided as [P(0), P(1)], use P(1)
            y_proba = y_proba[:, 1]
            
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

def calculate_per_class_metrics(y_true, y_pred, class_labels=None):
    """
    Calculate performance metrics for each class.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_labels (list): Optional list of class names
        
    Returns:
        DataFrame: DataFrame with per-class metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract per-class metrics
    class_metrics = []
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            metrics['class'] = class_name
            class_metrics.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(class_metrics)
    
    # Rename class labels if provided
    if class_labels is not None:
        label_map = {str(i): label for i, label in enumerate(class_labels)}
        df['class'] = df['class'].map(lambda x: label_map.get(x, x))
    
    return df

def calculate_confusion_matrix_stats(cm, class_labels=None):
    """
    Calculate additional statistics from confusion matrix.
    
    Args:
        cm (array): Confusion matrix
        class_labels (list): Optional list of class names
        
    Returns:
        DataFrame: DataFrame with statistics
    """
    n_classes = cm.shape[0]
    
    # Initialize results
    stats = []
    
    # Calculate per-class statistics
    for i in range(n_classes):
        # True positives, false positives, false negatives, true negatives
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store results
        class_name = class_labels[i] if class_labels is not None else f"Class {i}"
        stats.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'support': np.sum(cm[i, :])
        })
    
    return pd.DataFrame(stats)