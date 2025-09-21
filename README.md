# Machine Learning Projects Repository

This repository contains multiple machine learning projects with standardized project structures. Each project follows best practices for organizing ML code, making them easier to understand, maintain, and scale.

## Projects

- **exoplanet-transit-detection**: Detection of exoplanet transits using time series data from Kepler/TESS missions.
- **stellar-classification**: Classification of stars based on their spectral and photometric properties.

## Recent Updates

- **Flexible FITS file handling**: Added support for multiple flux column naming conventions (`PDCSAP_FLUX`, `FLUX`) in the shared utilities
- **Improved error handling**: Better handling of errors when processing FITS files
- **New example notebook**: Added `fits_handling_example.ipynb` to demonstrate the flexible FITS loading

## Repository Structure

```
practice1/
│
├── projects/                     # All ML projects
│   ├── exoplanet-transit-detection/  # Project 1
│   └── stellar-classification/       # Project 2 
│
├── shared/                       # Shared utilities across projects
│   ├── data_utils/               # Common data processing utilities
│   │   ├── data_processing.py    # General data processing functions
│   │   └── fits_loader.py        # FITS file handling utilities
│   ├── visualization/            # Common visualization code
│   └── evaluation/               # Common model evaluation metrics
│
└── notebooks/                    # Exploratory notebooks not specific to any project
    ├── explore_kepler.ipynb      # Kepler data exploration
    └── fits_handling_example.ipynb # Demo of FITS file handling
```

## Getting Started

Each project has its own README with setup instructions and documentation.

## Contributing

Please follow the project structure when adding new code or projects.
