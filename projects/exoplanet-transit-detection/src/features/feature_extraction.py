"""
Feature extraction for transit detection from light curves.
"""

import numpy as np
from astropy.timeseries import LombScargle

def extract_period_features(time, flux):
    """
    Extract period-related features using Lomb-Scargle periodogram.
    
    Args:
        time (array): Time array
        flux (array): Flux array
        
    Returns:
        tuple: (best_period, best_power) - the dominant period and its power
    """
    try:
        ls = LombScargle(time, flux)
        freq, power = ls.autopower()
        best_period = 1 / freq[np.argmax(power)]
        best_power = np.max(power)
        return best_period, best_power
    except Exception:
        return 0, 0

def extract_shape_features(local_flux):
    """
    Extract shape features from a localized flux window.
    
    Args:
        local_flux (array): Localized flux array around a potential transit
        
    Returns:
        tuple: (depth, width, symmetry) features of the dip
    """
    if len(local_flux) < 3:
        return 0, 0, 0
        
    depth = 1.0 - np.min(local_flux)
    width = np.sum(local_flux < (1 - depth/2))  # number of points below half-depth
    
    # symmetry: compare first half vs second half of dip
    if len(local_flux) > 4:
        symmetry = np.corrcoef(
            local_flux[:len(local_flux)//2],
            local_flux[len(local_flux)//2:]
        )[0, 1]
    else:
        symmetry = 0
        
    return depth, width, symmetry

def extract_window_features(local_flux, global_flux_mean=None):
    """
    Extract features from a window of flux values.
    
    Args:
        local_flux (array): Window of flux values
        global_flux_mean (float): Mean of the entire flux series (optional)
        
    Returns:
        tuple: (local_std, local_slope, window_ratio)
    """
    if len(local_flux) < 2:
        return 0, 0, 1
        
    # Standard deviation
    local_std = np.std(local_flux)
    
    # Linear slope
    local_slope = np.polyfit(range(len(local_flux)), local_flux, 1)[0]
    
    # Window ratio (if global_flux_mean provided)
    window_ratio = 1.0
    if global_flux_mean is not None:
        window_ratio = np.mean(local_flux) / global_flux_mean
        
    return local_std, local_slope, window_ratio

def calculate_snr(depth, local_std):
    """
    Calculate signal-to-noise ratio.
    
    Args:
        depth (float): Transit depth
        local_std (float): Local standard deviation
        
    Returns:
        float: Signal-to-noise ratio
    """
    return depth / (local_std + 1e-6)  # Add small epsilon to prevent division by zero

def identify_potential_transits(flux_norm, percentile=5):
    """
    Identify potential transit points using percentile threshold.
    
    Args:
        flux_norm (array): Normalized flux array
        percentile (float): Percentile threshold (default: 5)
        
    Returns:
        tuple: (threshold, dip_mask, dip_indices)
    """
    threshold = np.percentile(flux_norm, percentile)
    dip_mask = flux_norm < threshold
    dip_indices = np.where(dip_mask)[0]
    return threshold, dip_mask, dip_indices