"""
Neural Network model for transit detection.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def create_neural_network_model(input_shape, hidden_units=(16, 8), dropout_rates=(0.3, 0.2)):
    """
    Create a neural network model for transit detection.
    
    Args:
        input_shape (tuple): Shape of input features
        hidden_units (tuple): Number of units in hidden layers
        dropout_rates (tuple): Dropout rates for hidden layers
        
    Returns:
        Model: Keras model
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Define the model architecture
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=input_shape))
    
    # Hidden layers
    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, activation='relu'))
        if i < len(dropout_rates):
            model.add(layers.Dropout(dropout_rates[i]))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_neural_network(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train neural network model with early stopping.
    
    Args:
        model (Model): Keras model
        X_train (array): Training features
        y_train (array): Training labels
        epochs (int): Maximum number of epochs
        batch_size (int): Batch size
        validation_split (float): Fraction of data for validation
        
    Returns:
        tuple: (trained_model, history)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Calculate class weights
    class_weight = {
        0: 1.0,
        1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    }
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history, scaler

def evaluate_neural_network(model, X_test, y_test, scaler):
    """
    Evaluate neural network on test data.
    
    Args:
        model (Model): Trained Keras model
        X_test (array): Test features
        y_test (array): Test labels
        scaler (StandardScaler): Fitted feature scaler
        
    Returns:
        dict: Evaluation metrics
    """
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    return {
        'test_predictions': y_pred,
        'test_probabilities': y_pred_prob,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }