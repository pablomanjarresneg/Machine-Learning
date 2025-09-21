"""
Common visualization utilities for all projects.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

def set_plotting_style():
    """
    Set a consistent plotting style for all visualizations.
    """
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_feature_distributions(df, features=None, n_cols=3, figsize=(15, 12)):
    """
    Plot distributions of multiple features.
    
    Args:
        df (DataFrame): Input DataFrame
        features (list): List of feature names to plot (if None, use all)
        n_cols (int): Number of columns in subplot grid
        figsize (tuple): Figure size
    """
    # Select features to plot
    if features is None:
        features = df.select_dtypes(include=np.number).columns
    
    # Calculate number of rows needed
    n_rows = int(np.ceil(len(features) / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.histplot(df[feature], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, features=None, figsize=(12, 10)):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df (DataFrame): Input DataFrame
        features (list): List of feature names to include (if None, use all numeric)
        figsize (tuple): Figure size
    """
    # Select features to include
    if features is None:
        features = df.select_dtypes(include=np.number).columns
    
    # Calculate correlation matrix
    corr = df[features].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create plot
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), normalize=True):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        class_names (list): Names of classes
        figsize (tuple): Figure size
        normalize (bool): Whether to normalize by row
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Set class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Create plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm_display, annot=cm, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_curves(models_with_predictions, y_true):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_with_predictions (list): List of tuples (model_name, y_proba)
        y_true (array): True labels
    """
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Plot ROC curve for each model
    for model_name, y_proba in models_with_predictions:
        # For multi-class probabilities, use the second column (prob of class 1)
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_precision_recall_curves(models_with_predictions, y_true):
    """
    Plot precision-recall curves for multiple models.
    
    Args:
        models_with_predictions (list): List of tuples (model_name, y_proba)
        y_true (array): True labels
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate baseline
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='r', linestyle='--', 
               label=f'Baseline (class frequency = {baseline:.3f})')
    
    # Plot precision-recall curve for each model
    for model_name, y_proba in models_with_predictions:
        # For multi-class probabilities, use the second column (prob of class 1)
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            y_proba = y_proba[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        # Plot
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_learning_curve(train_sizes, train_scores, test_scores):
    """
    Plot learning curve from cross-validation results.
    
    Args:
        train_sizes (array): Training set sizes
        train_scores (array): Training scores for each size
        test_scores (array): Validation scores for each size
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation score')
    
    # Plot standard deviation bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                    alpha=0.1, color='green')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()