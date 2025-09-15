# 📊 Data handling & math
import numpy as np
import pandas as pd
from scipy import stats, signal

# 📈 Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ⚙️ Machine Learning & Deep Learning
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# 🤖 Classical ML & preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 🛠️ Utilities
from tqdm import tqdm   # progress bars
import os               # file paths
import warnings         # suppress warnings

#Astronomy-specific only if you use FITS/light curve data
from astropy.io import fits
from astropy.timeseries import TimeSeries
