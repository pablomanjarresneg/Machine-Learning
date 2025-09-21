"""
Neural Network model for stellar classification.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def create_nn_model(input_shape, num_classes, hidden_layers=(64, 32)):
    """
    Create a neural network model for stellar classification.
    
    Args:
        input_shape (tuple): Shape of input features
        num_classes (int): Number of classes
        hidden_layers (tuple): Number of units in hidden layers
        
    Returns:
        Model: Keras model
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        # Define the model architecture
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Hidden layers
        for units in hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
        
        # Output layer
        if num_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            # Multi-class classification
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    except ImportError:
        print("TensorFlow not available. Cannot create neural network model.")
        return None

def train_nn_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
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
    import tensorflow as tf
    from tensorflow import keras
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train the model
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history

def evaluate_nn_model(model, X_test, y_test):
    """
    Evaluate neural network on test data.
    
    Args:
        model (Model): Trained Keras model
        X_test (array): Test features
        y_test (array): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Scale features
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Calculate test accuracy
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Neural Network Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    if model.output_shape[-1] == 1:  # Binary classification
        y_pred_prob = model.predict(X_test_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_proba = np.hstack((1 - y_pred_prob, y_pred_prob))  # Convert to [P(0), P(1)] format
    else:  # Multi-class classification
        y_proba = model.predict(X_test_scaled)
        y_pred = np.argmax(y_proba, axis=1)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_predictions': y_pred,
        'test_probabilities': y_proba,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }