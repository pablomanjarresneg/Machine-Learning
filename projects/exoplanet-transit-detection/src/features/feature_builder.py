"""
Feature engineering and dataset preparation for transit detection models.
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample

def build_dataset_from_lightcurves(time_clean, flux_norm, dip_indices, dip_mask):
    """
    Build a dataset from light curve data for transit detection.
    
    Args:
        time_clean (array): Cleaned time array
        flux_norm (array): Normalized flux array
        dip_indices (array): Indices of potential transit points
        dip_mask (array): Boolean mask of potential transits
        
    Returns:
        tuple: (all_features, all_labels) for machine learning
    """
    from .feature_extraction import (
        extract_period_features, 
        extract_shape_features, 
        extract_window_features, 
        calculate_snr
    )
    
    # Calculate period features once for the entire lightcurve
    best_period, best_power = extract_period_features(time_clean, flux_norm)
    
    features = []
    labels = []
    
    # Get features for transit candidates (dips)
    for idx in dip_indices:
        # Extract a window around the dip
        window_start = max(0, idx-20)
        window_end = min(len(flux_norm), idx+20)
        local_flux = flux_norm[window_start:window_end]
        
        # Calculate shape features
        depth, width, symmetry = extract_shape_features(local_flux)
        
        # Calculate additional features
        local_std, local_slope, window_ratio = extract_window_features(
            local_flux, np.mean(flux_norm)
        )
        snr = calculate_snr(depth, local_std)
        
        # Criteria for transit identification
        is_likely_transit = (
            width > 2 and 
            symmetry > 0.3 and
            snr > 3 and
            local_slope < 0.02
        )
        
        features.append([
            local_std, width, symmetry, local_slope,
            window_ratio, snr, best_power
        ])
        labels.append(1 if is_likely_transit else 0)
    
    # Add non-transit baseline samples
    non_dip_mask = ~dip_mask
    normal_indices = np.random.choice(
        np.where(non_dip_mask)[0],
        min(len(dip_indices), sum(non_dip_mask)),
        replace=False
    )
    
    for idx in normal_indices:
        window_start = max(0, idx-20)
        window_end = min(len(flux_norm), idx+20)
        local_flux = flux_norm[window_start:window_end]
        
        depth, width, symmetry = extract_shape_features(local_flux)
        local_std, local_slope, window_ratio = extract_window_features(
            local_flux, np.mean(flux_norm)
        )
        snr = calculate_snr(depth, local_std)
        
        features.append([
            local_std, width, symmetry, local_slope,
            window_ratio, snr, best_power
        ])
        labels.append(0)  # Definitely not a transit
    
    return features, labels

def create_dataframe(features, labels, source_files=None):
    """
    Create a pandas DataFrame from features and labels.
    
    Args:
        features (list): List of feature vectors
        labels (list): List of labels
        source_files (list): Optional list of source file paths
        
    Returns:
        DataFrame: Pandas DataFrame with features and labels
    """
    df = pd.DataFrame(features, columns=[
        "local_noise", "width", "symmetry", "local_slope",
        "window_ratio", "snr", "best_power"
    ])
    df["label"] = labels
    
    if source_files:
        df["source_file"] = source_files
    
    return df

def balance_dataset(X, y, strategy='upsample', random_state=42):
    """
    Balance dataset by upsampling minority class or downsampling majority class.
    
    Args:
        X (DataFrame): Feature DataFrame
        y (Series): Target Series
        strategy (str): 'upsample' or 'downsample'
        random_state (int): Random seed
        
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    df = X.copy()
    df['label'] = y
    
    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()
    
    majority = df[df.label == majority_class]
    minority = df[df.label == minority_class]
    
    if strategy == 'upsample':
        # Upsample minority class
        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=random_state
        )
        df_balanced = pd.concat([majority, minority_upsampled])
    else:
        # Downsample majority class
        majority_downsampled = resample(
            majority,
            replace=False,
            n_samples=len(minority),
            random_state=random_state
        )
        df_balanced = pd.concat([majority_downsampled, minority])
    
    return df_balanced.drop('label', axis=1), df_balanced['label']