"""
Flexible FITS file loading utilities for astronomical time series data.

This module provides functions to load light curves from FITS files
with support for different flux column naming conventions.
"""

import numpy as np
from astropy.io import fits


def load_flux_from_fits(file_path, flux_columns=['PDCSAP_FLUX', 'FLUX', 'SAP_FLUX']):
    """
    Universal loader for light curve data that tries multiple flux column names.
    
    Args:
        file_path (str): Path to the FITS file
        flux_columns (list): List of flux column names to try, in order of preference
    
    Returns:
        tuple: (time, flux, quality, flux_column_used) if successful, otherwise (None, None, None, None)
    """
    try:
        with fits.open(file_path) as hdul:
            # Get the data from the first extension
            data = hdul[1].data
            
            # Check if we have time data
            if 'TIME' not in data.names:
                return None, None, None, None
                
            time = data['TIME']
            
            # Try each flux column in order
            flux = None
            flux_column_used = None
            for column in flux_columns:
                if column in data.names:
                    flux = data[column]
                    flux_column_used = column
                    break
                    
            if flux is None:
                return None, None, None, None
                
            # Get quality flags if they exist
            quality = data['QUALITY'] if 'QUALITY' in data.names else np.zeros_like(time)
            
            # Clean the data
            mask = (~np.isnan(time)) & (~np.isnan(flux)) & (quality == 0)
            
            if np.sum(mask) < 10:  # Not enough valid points
                return None, None, None, None
                
            return time[mask], flux[mask], quality[mask], flux_column_used
            
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None, None, None


def normalize_flux(flux):
    """
    Normalize flux values by dividing by the median.
    
    Args:
        flux (numpy.ndarray): Flux values to normalize
        
    Returns:
        numpy.ndarray: Normalized flux values
    """
    if flux is None or len(flux) == 0:
        return None
        
    return flux / np.median(flux)


def get_fits_header_info(file_path):
    """
    Extract useful metadata from FITS header.
    
    Args:
        file_path (str): Path to the FITS file
        
    Returns:
        dict: Dictionary of relevant header information
    """
    try:
        with fits.open(file_path) as hdul:
            primary_header = hdul[0].header
            data_header = hdul[1].header
            
            # Extract commonly useful information
            info = {
                'TELESCOP': primary_header.get('TELESCOP', 'Unknown'),
                'OBJECT': primary_header.get('OBJECT', data_header.get('OBJECT', 'Unknown')),
                'RA': primary_header.get('RA_OBJ', data_header.get('RA_OBJ', None)),
                'DEC': primary_header.get('DEC_OBJ', data_header.get('DEC_OBJ', None)),
                'DATE-OBS': primary_header.get('DATE-OBS', data_header.get('DATE-OBS', None)),
                'EXPTIME': primary_header.get('EXPTIME', data_header.get('EXPTIME', None)),
            }
            
            # Add more telescope-specific information
            if info['TELESCOP'] == 'Kepler':
                info['KEPLERID'] = primary_header.get('KEPLERID', data_header.get('KEPLERID', None))
                info['QUARTER'] = primary_header.get('QUARTER', data_header.get('QUARTER', None))
            elif info['TELESCOP'] == 'TESS':
                info['TICID'] = primary_header.get('TICID', data_header.get('TICID', None))
                info['SECTOR'] = primary_header.get('SECTOR', data_header.get('SECTOR', None))
                
            return info
            
    except Exception as e:
        print(f"Error reading header from {file_path}: {str(e)}")
        return {}


def list_fits_columns(file_path):
    """
    List all column names available in a FITS file.
    
    Args:
        file_path (str): Path to the FITS file
        
    Returns:
        list: List of column names in the FITS file
    """
    try:
        with fits.open(file_path) as hdul:
            if len(hdul) > 1:
                return list(hdul[1].data.names)
            else:
                return []
    except Exception as e:
        print(f"Error reading columns from {file_path}: {str(e)}")
        return []