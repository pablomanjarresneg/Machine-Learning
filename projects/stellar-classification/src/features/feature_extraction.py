"""
Feature extraction for stellar classification.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def extract_color_indices(data, magnitude_columns):
    """
    Calculate color indices from magnitude columns.
    
    Args:
        data (DataFrame): Pandas DataFrame with magnitude data
        magnitude_columns (list): List of magnitude column names in order (e.g., ['U', 'B', 'V', 'R', 'I'])
        
    Returns:
        DataFrame: DataFrame with calculated color indices
    """
    data_copy = data.copy()
    
    # Create color indices (e.g., U-B, B-V, etc.)
    for i in range(len(magnitude_columns) - 1):
        col1 = magnitude_columns[i]
        col2 = magnitude_columns[i + 1]
        index_name = f"{col1}-{col2}"
        
        if col1 in data.columns and col2 in data.columns:
            data_copy[index_name] = data[col1] - data[col2]
    
    return data_copy

def calculate_derived_parameters(data):
    """
    Calculate derived astrophysical parameters from stellar data.
    
    Args:
        data (DataFrame): Pandas DataFrame with stellar data
        
    Returns:
        DataFrame: DataFrame with calculated parameters
    """
    data_copy = data.copy()
    
    # Calculate absolute magnitude if distance and apparent magnitude are available
    if 'distance' in data.columns and 'apparent_magnitude' in data.columns:
        data_copy['absolute_magnitude'] = data['apparent_magnitude'] - 5 * np.log10(data['distance'] / 10)
    
    # Calculate effective temperature from B-V color index (simplified formula)
    if 'B-V' in data.columns:
        data_copy['effective_temperature'] = 9500 / (1 + 0.93 * data['B-V'])
    
    # Calculate luminosity from absolute magnitude (simplified formula)
    if 'absolute_magnitude' in data.columns:
        # Solar absolute magnitude is 4.83
        data_copy['luminosity'] = 10 ** ((4.83 - data['absolute_magnitude']) / 2.5)
    
    return data_copy

def apply_feature_selection(X, y, n_features=10, method='selectk'):
    """
    Apply feature selection to identify most important features.
    
    Args:
        X (DataFrame): Feature DataFrame
        y (Series): Target Series
        n_features (int): Number of features to select
        method (str): Selection method ('selectk' or 'pca')
        
    Returns:
        tuple: (DataFrame with selected features, feature selector)
    """
    if method.lower() == 'selectk':
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get names of selected features
        selected_indices = selector.get_support(indices=True)
        selected_columns = X.columns[selected_indices]
        
        return pd.DataFrame(X_selected, columns=selected_columns), selector
    
    elif method.lower() == 'pca':
        # Standardize data first for PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=min(n_features, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create column names for PCA components
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        return pd.DataFrame(X_pca, columns=pca_columns), pca
    
    else:
        raise ValueError(f"Unsupported feature selection method: {method}")

def analyze_feature_importance(X, y):
    """
    Analyze feature importance using different methods.
    
    Args:
        X (DataFrame): Feature DataFrame
        y (Series): Target Series
        
    Returns:
        DataFrame: DataFrame with feature importance scores
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # ANOVA F-value
    f_scores, p_values = f_classif(X, y)
    
    # Create DataFrame with importance scores
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf_importance,
        'f_score': f_scores,
        'p_value': p_values
    })
    
    # Sort by Random Forest importance
    importance_df = importance_df.sort_values('rf_importance', ascending=False)
    
    return importance_df