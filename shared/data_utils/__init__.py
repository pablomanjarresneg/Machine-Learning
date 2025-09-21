"""
Data utilities for astronomical data processing.
"""

from .data_processing import *
# Import the new fits_loader module
from .fits_loader import (
    load_flux_from_fits, 
    normalize_flux, 
    get_fits_header_info,
    list_fits_columns
)

__all__ = [
    'load_flux_from_fits',
    'normalize_flux',
    'get_fits_header_info',
    'list_fits_columns'
]