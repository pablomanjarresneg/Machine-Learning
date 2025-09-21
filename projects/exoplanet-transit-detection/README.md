# Exoplanet Transit Detection

This project focuses on detecting exoplanet transits in light curve data from Kepler and TESS missions using machine learning techniques.

## Project Overview

Exoplanet transit detection involves identifying the small dips in stellar brightness that occur when a planet passes in front of its host star. This project uses machine learning models to automatically detect these transit events in time-series light curve data.

## Data

The project uses light curve data from:
- Kepler mission
- TESS mission

Data is stored in FITS format containing time series brightness measurements.

## Approach

1. **Data Processing**: Clean light curve data by removing NaN values and normalizing flux values
2. **Feature Engineering**: Extract features from light curves including:
   - Transit depth, width, and symmetry
   - Signal-to-noise ratio
   - Period information
3. **Model Training**: 
   - Random Forest classifier
   - Neural Network model
4. **Evaluation**: Metrics including precision, recall, and ROC curves

## Directory Structure

```
exoplanet-transit-detection/
├── data/                  # Data files
│   ├── raw/               # Original FITS files
│   └── processed/         # Processed data and features
│
├── notebooks/             # Jupyter notebooks
│   ├── explore_kepler.ipynb     # Exploratory analysis of Kepler data
│   └── explore_tess.ipynb       # Exploratory analysis of TESS data
│
├── src/                   # Source code
│   ├── data/              # Data processing code
│   ├── features/          # Feature engineering code
│   ├── models/            # Model training code
│   └── visualization/     # Visualization code
│
└── models/                # Saved model files
```

## Getting Started

1. Install requirements:
```
pip install -r requirements.txt
```

2. Process raw data:
```
python src/data/make_dataset.py
```

3. Train models:
```
python src/models/train_model.py
```

## Results

The models achieve over 90% precision in detecting transit events, with key features being transit shape, symmetry and signal-to-noise ratio.