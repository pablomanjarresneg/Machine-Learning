"""
Visualization utilities for stellar classification.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_class_distribution(y, title="Class Distribution"):
    """
    Plot distribution of classes.
    
    Args:
        y (array): Array of class labels
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Count classes
    class_counts = pd.Series(y).value_counts().sort_index()
    
    # Create bar plot
    ax = sns.barplot(x=class_counts.index, y=class_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(class_counts.values):
        ax.text(i, count + 5, str(count), ha='center')
    
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_feature_correlation(X, n_features=10, title="Feature Correlation Matrix"):
    """
    Plot correlation matrix of features.
    
    Args:
        X (DataFrame): Feature DataFrame
        n_features (int): Number of features to include
        title (str): Plot title
    """
    # Select subset of features if there are many
    if X.shape[1] > n_features:
        # Select features based on variance
        variances = X.var().sort_values(ascending=False)
        selected_features = variances.index[:n_features]
        X_subset = X[selected_features]
    else:
        X_subset = X
    
    # Calculate correlation matrix
    corr = X_subset.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importances(feature_importances, feature_names):
    """
    Plot feature importances from a model.
    
    Args:
        feature_importances (array): Feature importance scores
        feature_names (list): List of feature names
    """
    # Convert to array if needed
    if isinstance(feature_importances, list):
        feature_importances = np.array(feature_importances)
    
    # Get indices of top features
    indices = np.argsort(feature_importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(feature_importances)), feature_importances[indices], align='center')
    plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Print ordered feature importances
    print("Feature ranking:")
    for i in range(len(feature_importances)):
        print(f"{i+1}. {feature_names[indices[i]]}: {feature_importances[indices[i]]:.4f}")

def plot_confusion_matrix(cm, title="Confusion Matrix", class_names=None):
    """
    Plot a confusion matrix.
    
    Args:
        cm (array): Confusion matrix
        title (str): Plot title
        class_names (list): Optional list of class names
    """
    plt.figure(figsize=(10, 8))
    
    # If class names not provided, use default labels
    if class_names is None:
        n_classes = cm.shape[0]
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_nn_history(history):
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

def plot_feature_distributions(X, y, feature_names=None, n_features=5):
    """
    Plot distributions of top features by class.
    
    Args:
        X (DataFrame): Feature DataFrame
        y (array): Class labels
        feature_names (list): Names of features to plot (if None, select by variance)
        n_features (int): Number of features to plot if feature_names is None
    """
    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Add class column
    data = X.copy()
    data['class'] = y
    
    # Select features to plot
    if feature_names is None:
        # Select top features by variance
        variances = X.var().sort_values(ascending=False)
        feature_names = variances.index[:n_features]
    
    # Create subplots
    n_plots = len(feature_names)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    
    # Ensure axes is always a list
    if n_plots == 1:
        axes = [axes]
    
    # Plot each feature
    for i, feature in enumerate(feature_names):
        sns.violinplot(x='class', y=feature, data=data, ax=axes[i])
        axes[i].set_title(f'Distribution of {feature} by Class')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel(feature)
    
    plt.tight_layout()
    plt.show()