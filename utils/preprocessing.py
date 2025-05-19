import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def load_transaction_data(file_path):
    """
    Load and preprocess financial transaction data.
    
    Args:
        file_path (str): Path to the transaction data file
        
    Returns:
        pd.DataFrame: Preprocessed transaction data
    """
    # Read the data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df

def extract_features(df):
    """
    Extract relevant features for anomaly detection.
    
    Args:
        df (pd.DataFrame): Raw transaction data
        
    Returns:
        pd.DataFrame: Feature matrix
    """
    features = pd.DataFrame()
    
    # Amount-based features
    features['amount'] = df['amount']
    features['amount_log'] = np.log1p(df['amount'])
    
    # Time-based features if available
    if 'hour' in df.columns:
        features['hour'] = df['hour']
    if 'day_of_week' in df.columns:
        features['day_of_week'] = df['day_of_week']
    
    # Categorical features (if any)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'timestamp':  # Skip timestamp column
            features[f'{col}_encoded'] = pd.Categorical(df[col]).codes
    
    return features

def normalize_features(features):
    """
    Normalize features using StandardScaler.
    
    Args:
        features (pd.DataFrame): Feature matrix
        
    Returns:
        np.ndarray: Normalized features
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def create_sequence_features(df, sequence_length=10):
    """
    Create sequence-based features for time series analysis.
    
    Args:
        df (pd.DataFrame): Transaction data
        sequence_length (int): Length of sequence to consider
        
    Returns:
        np.ndarray: Sequence features
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create sequences
    sequences = []
    for i in range(len(df) - sequence_length + 1):
        sequence = df.iloc[i:i+sequence_length]
        sequences.append(sequence['amount'].values)
    
    return np.array(sequences) 