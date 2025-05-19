import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import joblib
import os

class AnomalyDetector:
    def __init__(self, method='isolation_forest', contamination=0.1):
        """
        Initialize the anomaly detector with specified method and parameters.
        
        Args:
            method (str): 'isolation_forest' or 'autoencoder'
            contamination (float): Expected proportion of anomalies in the data
        """
        self.method = method
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = None
        
        if method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == 'autoencoder':
            pass  # Will initialize during fit
        else:
            raise ValueError("Method must be either 'isolation_forest' or 'autoencoder'")
    
    def _build_autoencoder(self, input_dim):
        """Build and return an autoencoder model."""
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(encoder_input)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder
        autoencoder = Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def fit(self, X):
        """
        Fit the anomaly detection model to the data.
        
        Args:
            X (pd.DataFrame or np.ndarray): Training data
        """
        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'isolation_forest':
            self.model.fit(X_scaled)
        else:  # autoencoder
            # Initialize autoencoder with correct input dimension
            self.model = self._build_autoencoder(X.shape[1])
            self.model.fit(
                X_scaled, X_scaled,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
    
    def predict(self, X):
        """
        Predict anomalies in the data.
        
        Args:
            X (pd.DataFrame or np.ndarray): Data to predict
            
        Returns:
            np.ndarray: 1 for normal, -1 for anomaly
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'isolation_forest':
            return self.model.predict(X_scaled)
        else:  # autoencoder
            # Calculate reconstruction error
            X_pred = self.model.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
            threshold = np.percentile(mse, (1 - self.contamination) * 100)
            return np.where(mse > threshold, -1, 1)
    
    def save_model(self, path):
        """Save the model and scaler to disk."""
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.method == 'isolation_forest':
            joblib.dump(self.model, os.path.join(path, 'model.joblib'))
        else:
            self.model.save(os.path.join(path, 'model.h5'))
        
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
    
    def load_model(self, path):
        """Load the model and scaler from disk."""
        if self.method == 'isolation_forest':
            self.model = joblib.load(os.path.join(path, 'model.joblib'))
        else:
            self.model = tf.keras.models.load_model(os.path.join(path, 'model.h5'))
        
        self.scaler = joblib.load(os.path.join(path, 'scaler.joblib')) 