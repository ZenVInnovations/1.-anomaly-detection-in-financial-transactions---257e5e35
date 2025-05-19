import pandas as pd
import numpy as np
from anomaly_detector import AnomalyDetector
from utils.preprocessing import load_transaction_data, extract_features, normalize_features
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_data(n_samples=1000):
    """Generate sample transaction data for demonstration."""
    np.random.seed(42)
    
    # Generate normal transactions
    normal_amounts = np.random.lognormal(mean=4, sigma=1, size=n_samples)
    normal_timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # Generate some anomalies
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    normal_amounts[anomaly_indices] *= np.random.uniform(5, 10, size=len(anomaly_indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': normal_timestamps,
        'amount': normal_amounts,
        'transaction_type': np.random.choice(['purchase', 'transfer', 'withdrawal'], size=n_samples)
    })
    
    return df

def main():
    # Generate sample data
    print("Generating sample transaction data...")
    df = generate_sample_data()
    
    # Save sample data
    df.to_csv('data/sample_transactions.csv', index=False)
    
    # Load and preprocess data
    print("Preprocessing data...")
    df = load_transaction_data('data/sample_transactions.csv')
    features = extract_features(df)
    normalized_features = normalize_features(features)
    
    # Initialize and train Isolation Forest model
    print("Training Isolation Forest model...")
    if_model = AnomalyDetector(method='isolation_forest', contamination=0.05)
    if_model.fit(normalized_features)
    
    # Initialize and train Autoencoder model
    print("Training Autoencoder model...")
    ae_model = AnomalyDetector(method='autoencoder', contamination=0.05)
    ae_model.fit(normalized_features)
    
    # Make predictions
    if_predictions = if_model.predict(normalized_features)
    ae_predictions = ae_model.predict(normalized_features)
    
    # Add predictions to original dataframe
    df['if_anomaly'] = if_predictions
    df['ae_anomaly'] = ae_predictions
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Transaction amounts with anomalies highlighted
    plt.subplot(2, 1, 1)
    plt.scatter(df.index, df['amount'], c='blue', alpha=0.5, label='Normal')
    plt.scatter(df[df['if_anomaly'] == -1].index, 
                df[df['if_anomaly'] == -1]['amount'],
                c='red', label='Anomaly (IF)')
    plt.title('Transaction Amounts with Anomalies (Isolation Forest)')
    plt.xlabel('Transaction Index')
    plt.ylabel('Amount')
    plt.legend()
    
    # Plot 2: Transaction amounts with autoencoder anomalies
    plt.subplot(2, 1, 2)
    plt.scatter(df.index, df['amount'], c='blue', alpha=0.5, label='Normal')
    plt.scatter(df[df['ae_anomaly'] == -1].index,
                df[df['ae_anomaly'] == -1]['amount'],
                c='red', label='Anomaly (AE)')
    plt.title('Transaction Amounts with Anomalies (Autoencoder)')
    plt.xlabel('Transaction Index')
    plt.ylabel('Amount')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/anomaly_detection_results.png')
    plt.close()
    
    # Print summary statistics
    print("\nAnomaly Detection Results:")
    print(f"Total transactions: {len(df)}")
    print(f"Isolation Forest detected {sum(df['if_anomaly'] == -1)} anomalies")
    print(f"Autoencoder detected {sum(df['ae_anomaly'] == -1)} anomalies")
    
    # Save models
    if_model.save_model('models/isolation_forest')
    ae_model.save_model('models/autoencoder')
    
    print("\nModels saved to 'models/' directory")
    print("Results visualization saved to 'data/anomaly_detection_results.png'")

if __name__ == "__main__":
    main() 