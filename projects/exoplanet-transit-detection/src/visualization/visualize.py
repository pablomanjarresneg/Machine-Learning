"""
Visualization utilities for light curves and model evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, roc_curve

def plot_light_curve(time, flux, title="Light Curve"):
    """
    Plot a light curve.
    
    Args:
        time (array): Time array
        flux (array): Flux array
        title (str): Plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time, flux, ".", markersize=2)
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.title(title)
    plt.show()

def plot_light_curve_with_transits(time, flux, dip_mask, title="Light Curve with Transits"):
    """
    Plot a light curve with highlighted transit points.
    
    Args:
        time (array): Time array
        flux (array): Flux array
        dip_mask (array): Boolean mask of transit points
        title (str): Plot title
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time, flux, ".", alpha=0.5, label="Flux", markersize=2)
    plt.plot(time[dip_mask], flux[dip_mask], "ro", label="Transit candidates", markersize=4)
    plt.xlabel("Time (days)")
    plt.ylabel("Normalized Flux")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_folded_light_curve(time, flux, period, title="Phase-Folded Light Curve"):
    """
    Plot a phase-folded light curve.
    
    Args:
        time (array): Time array
        flux (array): Flux array
        period (float): Folding period in days
        title (str): Plot title
    """
    phase = (time % period) / period
    plt.figure(figsize=(10, 5))
    plt.plot(phase, flux, ".", markersize=2, alpha=0.5)
    plt.xlabel(f"Phase (period = {period:.2f} days)")
    plt.ylabel("Normalized Flux")
    plt.title(title)
    plt.show()

def plot_feature_importances(feature_importances, feature_names):
    """
    Plot feature importances from a model.
    
    Args:
        feature_importances (array): Feature importance scores
        feature_names (list): List of feature names
    """
    indices = np.argsort(feature_importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(feature_importances)), feature_importances[indices], align='center')
    plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Print ordered feature importances
    print("Feature ranking:")
    for i in range(len(feature_importances)):
        print(f"{i+1}. {feature_names[indices[i]]}: {feature_importances[indices[i]]:.4f}")

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plot a confusion matrix.
    
    Args:
        cm (array): Confusion matrix
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Transit', 'Transit'],
                yticklabels=['Non-Transit', 'Transit'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.show()
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall/Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    """
    Plot precision-recall curve.
    
    Args:
        y_true (array): True labels
        y_proba (array): Predicted probabilities
        title (str): Plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()

def plot_neural_network_history(history):
    """
    Plot neural network training history.
    
    Args:
        history (History): Keras training history
    """
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.show()