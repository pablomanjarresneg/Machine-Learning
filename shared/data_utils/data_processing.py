"""
Common data processing utilities for all projects.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

def load_csv_data(file_path, **kwargs):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        DataFrame: Pandas DataFrame
    """
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def handle_missing_values(df, strategy='median', categorical_strategy='most_frequent'):
    """
    Handle missing values in a DataFrame.
    
    Args:
        df (DataFrame): Input DataFrame
        strategy (str): Imputation strategy for numeric columns
        categorical_strategy (str): Imputation strategy for categorical columns
        
    Returns:
        DataFrame: DataFrame with imputed values
    """
    # Make a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Split numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    
    # Impute numeric columns
    if numeric_cols.any():
        imputer = SimpleImputer(strategy=strategy)
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Impute categorical columns
    if categorical_cols.any():
        imputer = SimpleImputer(strategy=categorical_strategy)
        df_imputed[categorical_cols] = imputer.fit_transform(df[categorical_cols])
    
    return df_imputed

def scale_features(X, method='standard', return_scaler=False):
    """
    Scale features using different methods.
    
    Args:
        X (array or DataFrame): Features to scale
        method (str): Scaling method ('standard', 'minmax', 'robust')
        return_scaler (bool): Whether to return the scaler
        
    Returns:
        array or tuple: Scaled features, and optionally the scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_scaled = scaler.fit_transform(X)
    
    if return_scaler:
        return X_scaled, scaler
    else:
        return X_scaled

def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Args:
        X (array or DataFrame): Features
        y (array or Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def check_class_balance(y):
    """
    Check class balance in classification data.
    
    Args:
        y (array or Series): Target variable
        
    Returns:
        dict: Class distribution statistics
    """
    # Convert to pandas Series if not already
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Count classes
    class_counts = y.value_counts()
    total = len(y)
    
    # Calculate proportions
    class_proportions = class_counts / total
    
    # Calculate imbalance ratio (max count / min count)
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    return {
        'counts': class_counts,
        'proportions': class_proportions,
        'imbalance_ratio': imbalance_ratio,
        'is_balanced': imbalance_ratio < 2  # Rule of thumb for "balanced"
    }