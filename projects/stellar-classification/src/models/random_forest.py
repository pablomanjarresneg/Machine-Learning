"""
Random Forest model for stellar classification.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np

def create_rf_pipeline(n_estimators=100, max_depth=10, class_weight='balanced'):
    """
    Create a pipeline with scaling and Random Forest classifier.
    
    Args:
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees
        class_weight (str or dict): Class weights
        
    Returns:
        Pipeline: Scikit-learn pipeline
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),  
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            class_weight=class_weight,
            random_state=42
        ))
    ])
    
    return pipe

def evaluate_rf_model(pipe, X_train, y_train, X_test, y_test, cv=5):
    """
    Evaluate Random Forest model with cross-validation and test set.
    
    Args:
        pipe (Pipeline): Scikit-learn pipeline
        X_train (array): Training features
        y_train (array): Training labels
        X_test (array): Test features
        y_test (array): Test labels
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Evaluation metrics
    """
    # Cross-validation
    scores = cross_val_score(pipe, X_train, y_train, cv=cv)
    cv_score = {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std()
    }
    
    # Train final model
    pipe.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipe.predict(X_test)
    
    # Get class probabilities if possible
    if hasattr(pipe, 'predict_proba'):
        y_proba = pipe.predict_proba(X_test)
    else:
        y_proba = None
    
    # Get feature importances
    feature_importances = pipe.named_steps['clf'].feature_importances_
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Test Accuracy: {test_accuracy:.4f}")
    print(f"Cross-validation Accuracy: {cv_score['mean_accuracy']:.4f} ± {cv_score['std_accuracy']:.4f}")
    
    return {
        'cv_score': cv_score,
        'test_accuracy': test_accuracy,
        'test_predictions': y_pred,
        'test_probabilities': y_proba,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importances': feature_importances
    }