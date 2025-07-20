import numpy as np
import tensorflow as tf

class Predictors:
    def __init__(self, rf_model, nn_model):
        self.rf_model = rf_model
        self.nn_model = nn_model

    def predict_with_rf(self, X):
        if self.rf_model is None:
            raise ValueError("Random Forest model not trained.")
        return self.rf_model.predict(X)

    def predict_with_nn(self, X):
        if self.nn_model is None:
            raise ValueError("Neural Network model not trained.")
        # Ensure input is a NumPy array for TensorFlow
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        return self.nn_model.predict(X)

    def ensemble_predict(self, X, rf_weight=0.5, nn_weight=0.5):
        if self.rf_model is None or self.nn_model is None:
            raise ValueError("Both Random Forest and Neural Network models must be trained for ensemble prediction.")
        
        rf_predictions = self.predict_with_rf(X)
        nn_predictions = self.predict_with_nn(X)
        
        # Ensure nn_predictions is a 1D array if it's a 2D array with shape (n_samples, 1)
        if nn_predictions.ndim == 2 and nn_predictions.shape[1] == 1:
            nn_predictions = nn_predictions.flatten()

        return (rf_weight * rf_predictions) + (nn_weight * nn_predictions)
