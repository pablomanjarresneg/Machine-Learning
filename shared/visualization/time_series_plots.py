"""
Time series visualization utilities for all projects.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_time_series(time, values, title="Time Series", xlabel="Time", ylabel="Value"):
    """
    Plot a time series.
    
    Args:
        time (array): Time points
        values (array): Values at each time point
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_multiple_time_series(time, series_dict, title="Multiple Time Series", xlabel="Time", ylabel="Value"):
    """
    Plot multiple time series on the same graph.
    
    Args:
        time (array): Time points
        series_dict (dict): Dictionary of series_name: values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    plt.figure(figsize=(12, 6))
    
    for name, values in series_dict.items():
        plt.plot(time, values, label=name)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_folded_time_series(time, values, period, title="Folded Time Series"):
    """
    Plot a phase-folded time series.
    
    Args:
        time (array): Time points
        values (array): Values at each time point
        period (float): Folding period
        title (str): Plot title
    """
    # Calculate phase
    phase = (time % period) / period
    
    # Sort by phase
    idx = np.argsort(phase)
    phase_sorted = phase[idx]
    values_sorted = values[idx]
    
    plt.figure(figsize=(12, 6))
    plt.plot(phase_sorted, values_sorted, 'o-', markersize=3, alpha=0.7)
    plt.title(title)
    plt.xlabel(f"Phase (period = {period:.4f})")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def plot_autocorrelation(series, lags=40, title="Autocorrelation"):
    """
    Plot autocorrelation and partial autocorrelation functions.
    
    Args:
        series (array): Time series values
        lags (int): Number of lags to plot
        title (str): Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot ACF
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title(f"{title} - Autocorrelation")
    
    # Plot PACF
    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title(f"{title} - Partial Autocorrelation")
    
    plt.tight_layout()
    plt.show()

def plot_periodogram(time, values, title="Periodogram"):
    """
    Plot a periodogram to identify periodic signals.
    
    Args:
        time (array): Time points
        values (array): Values at each time point
        title (str): Plot title
    """
    try:
        from astropy.timeseries import LombScargle
        
        # Compute periodogram
        frequency, power = LombScargle(time, values).autopower()
        
        # Convert frequency to period
        period = 1 / frequency
        
        # Find the peak
        peak_idx = np.argmax(power)
        peak_period = period[peak_idx]
        
        plt.figure(figsize=(12, 6))
        plt.semilogx(period, power)
        plt.axvline(x=peak_period, color='r', linestyle='--', 
                   label=f'Peak period: {peak_period:.4f}')
        
        plt.title(title)
        plt.xlabel("Period")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return peak_period, power[peak_idx]
    
    except ImportError:
        print("astropy is required for periodogram analysis.")
        return None, None

def plot_seasonal_decomposition(df, column, period, model='additive'):
    """
    Plot seasonal decomposition of a time series.
    
    Args:
        df (DataFrame): DataFrame with datetime index
        column (str): Name of column to decompose
        period (int): Period for decomposition
        model (str): 'additive' or 'multiplicative'
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform decomposition
        result = seasonal_decompose(df[column], model=model, period=period)
        
        # Plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        
        result.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        ax1.set_xlabel('')
        
        result.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        ax2.set_xlabel('')
        
        result.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        ax3.set_xlabel('')
        
        result.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        plt.show()
        
        return result
    
    except ImportError:
        print("statsmodels is required for seasonal decomposition.")
        return None