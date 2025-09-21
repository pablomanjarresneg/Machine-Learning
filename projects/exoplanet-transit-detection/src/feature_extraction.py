"""
Feature extraction for exoplanet transit detection.

This module provides functions to extract features from light curves
that can be used to identify exoplanet transits.
"""

import numpy as np
from astropy.timeseries import LombScargle
import glob

# Import shared utilities
from shared.data_utils import load_flux_from_fits, normalize_flux


def period_features(time, flux):
    """
    Extract period-related features using Lomb-Scargle periodogram.
    
    Args:
        time (array): Time values
        flux (array): Flux values
        
    Returns:
        tuple: (best_period, best_power)
    """
    try:
        # Ensure we have enough data points
        if len(time) < 10:
            return 0, 0
            
        # Handle any NaN values
        valid_mask = ~np.isnan(time) & ~np.isnan(flux)
        if np.sum(valid_mask) < 10:
            return 0, 0
            
        time_valid = time[valid_mask]
        flux_valid = flux[valid_mask]
        
        # Run Lomb-Scargle
        ls = LombScargle(time_valid, flux_valid)
        freq, power = ls.autopower()
        
        # Check if we got valid results
        if len(freq) == 0 or np.all(np.isnan(power)):
            return 0, 0
            
        best_period = 1 / freq[np.argmax(power)]
        best_power = np.max(power)
        
        # Sanity check the period (sometimes we get unrealistic values)
        if best_period < 0.1 or best_period > 100:
            return 0, 0
            
        return best_period, best_power
    except Exception:
        return 0, 0


def shape_features(local_flux):
    """
    Extract shape-related features from a local flux window.
    
    Args:
        local_flux (array): Normalized flux values for a local window
        
    Returns:
        tuple: (depth, width, symmetry)
    """
    if len(local_flux) < 3:
        return 0, 0, 0
    
    depth = 1.0 - np.min(local_flux)
    width = np.sum(local_flux < (1 - depth/2))  # number of points below half-depth
    
    # symmetry: compare first half vs second half of dip
    symmetry = np.corrcoef(local_flux[:len(local_flux)//2],
                           local_flux[len(local_flux)//2:])[0, 1] if len(local_flux) > 4 else 0
    
    return depth, width, symmetry


def extract_transit_features(time, flux_norm, window_size=20, threshold_percentile=5):
    """
    Extract features for transit detection from a normalized light curve.
    
    Args:
        time (array): Time values
        flux_norm (array): Normalized flux values
        window_size (int): Size of window around potential transit
        threshold_percentile (float): Percentile threshold for identifying dips
        
    Returns:
        tuple: (features, is_transit_flags)
    """
    # Calculate period features once for the entire light curve
    best_period, best_power = period_features(time, flux_norm)
    
    # Find dips using percentile threshold
    threshold = np.percentile(flux_norm, threshold_percentile)
    dip_mask = flux_norm < threshold
    dip_indices = np.where(dip_mask)[0]
    
    features = []
    is_transit_flags = []
    
    # Process transit candidates (dips)
    for idx in dip_indices:
        # Extract a window around the dip
        window_start = max(0, idx - window_size)
        window_end = min(len(flux_norm), idx + window_size)
        local_flux = flux_norm[window_start:window_end]
        local_time = time[window_start:window_end]
        
        # Calculate shape features
        depth, width, symmetry = shape_features(local_flux)
        
        # Calculate additional features
        local_slope = np.polyfit(range(len(local_flux)), local_flux, 1)[0]
        local_std = np.std(local_flux)
        window_ratio = np.mean(local_flux) / np.mean(flux_norm)
        snr = depth / (local_std + 1e-6)
        
        # Assess if this is likely a transit
        is_likely_transit = (
            width > 2 and 
            symmetry > 0.3 and
            snr > 3 and
            local_slope < 0.02
        )
        
        # Collect features
        features.append([
            local_std, width, symmetry, local_slope,
            window_ratio, snr, best_power
        ])
        is_transit_flags.append(1 if is_likely_transit else 0)
    
    return features, is_transit_flags


def process_fits_files(directory_pattern, flux_columns=['PDCSAP_FLUX', 'FLUX'], threshold_percentile=5):
    """
    Process multiple FITS files and extract transit features.
    
    Args:
        directory_pattern (str): Glob pattern to match FITS files
        flux_columns (list): List of flux column names to try
        threshold_percentile (float): Percentile threshold for identifying dips
        
    Returns:
        tuple: (features_list, labels_list, sources_list, stats)
    """
    all_features = []
    all_labels = []
    all_sources = []
    stats = {'successful': 0, 'errors': 0, 'by_column': {}}
    
    # Initialize stats for each column type
    for column in flux_columns:
        stats['by_column'][column] = {'successful': 0, 'features': 0}
    
    # Process each file
    for path in glob.glob(directory_pattern, recursive=True):
        for flux_column in flux_columns:
            try:
                # Try to load with the current flux column
                time, flux, quality, column_used = load_flux_from_fits(path, [flux_column])
                
                if time is None or flux is None or column_used is None:
                    continue
                
                # Normalize the flux
                flux_norm = normalize_flux(flux)
                
                # Extract transit features
                features, labels = extract_transit_features(
                    time, flux_norm, threshold_percentile=threshold_percentile
                )
                
                if features and labels:
                    # Add to results
                    all_features.extend(features)
                    all_labels.extend(labels)
                    all_sources.extend([f"{path}:{column_used}"] * len(features))
                    
                    # Update statistics
                    stats['successful'] += 1
                    stats['by_column'][column_used]['successful'] += 1
                    stats['by_column'][column_used]['features'] += len(features)
                    
                    # Successfully processed with this column, move to next file
                    break
                    
            except Exception:
                # Count errors but continue to next file/column
                stats['errors'] += 1
    
    # Print summary statistics
    print(f"Processed FITS files: {stats['successful']} successful, {stats['errors']} with errors")
    for column, column_stats in stats['by_column'].items():
        if column_stats['successful'] > 0:
            print(f"  {column}: {column_stats['successful']} files, {column_stats['features']} features")
    
    return all_features, all_labels, all_sources, stats