# Stellar Classification

This project focuses on classifying different types of stars based on their spectral and photometric properties using machine learning techniques.

## Project Overview

Stellar classification is the categorization of stars based on their spectral characteristics. Traditional stellar classification uses the Morgan–Keenan (MK) system, with the main classes being O, B, A, F, G, K, and M. This project applies machine learning to automate this classification process using data from stellar catalogs.

### Features

- Data loading and preprocessing for stellar catalog data
- Feature extraction from stellar spectra and photometric measurements
- Machine learning models for multi-class stellar classification
- Evaluation metrics and visualization tools for model performance

## Directory Structure

```
stellar-classification/
│
├── data/                  # Data storage
│   ├── raw/               # Raw stellar catalog data
│   └── processed/         # Processed features and datasets
│
├── models/                # Saved model files
│
├── notebooks/             # Jupyter notebooks for exploration
│
└── src/                   # Source code
    ├── data/              # Data loading and processing
    ├── evaluation/        # Model evaluation tools
    ├── features/          # Feature extraction 
    ├── models/            # ML model definitions
    └── visualization/     # Visualization utilities
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, pandas, scikit-learn, tensorflow, astropy, matplotlib

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Place stellar catalog data in the `data/raw` directory
2. Run the data processing pipeline:
   ```
   python -m src.data.make_dataset
   ```
3. Train models:
   ```
   python -m src.models.train_model
   ```

## Models

The project includes several model types for stellar classification:

1. **Random Forest Classifier**: Ensemble learning approach
2. **Support Vector Machine**: For complex decision boundaries
3. **Neural Network**: Deep learning approach for spectral classification

## Features Used

- Effective temperature
- Surface gravity (log g)
- Metallicity
- Absolute magnitude
- Color indices (B-V, U-B, etc.)
- Spectral line strengths
- Proper motion

## Results

Model performance is evaluated on:
- Accuracy across different stellar classes
- Confusion matrix to identify misclassifications
- Precision, recall, and F1-score for each class
- Feature importance analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data from stellar catalogs and surveys
- Inspiration from astronomical classification methodologies