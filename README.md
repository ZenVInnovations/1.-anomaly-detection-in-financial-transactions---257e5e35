# Financial Transaction Anomaly Detection

This project implements advanced AI-driven anomaly detection algorithms to identify atypical patterns in financial transaction data. It uses both Isolation Forest and Autoencoder-based approaches to detect potential fraudulent activities or errors.

## Features

- Two different anomaly detection methods:
  - Isolation Forest: An unsupervised learning algorithm that isolates anomalies
  - Autoencoder: A neural network-based approach that learns normal patterns and detects deviations
- Comprehensive data preprocessing pipeline
- Feature engineering for financial transactions
- Visualization of detected anomalies
- Model persistence and loading capabilities

## Project Structure

```
financial_anomaly_detection/
├── data/                   # Directory for data files
├── models/                 # Directory for saved models
├── utils/                  # Utility functions
│   └── preprocessing.py    # Data preprocessing utilities
├── anomaly_detector.py     # Main anomaly detection implementation
├── example.py             # Example usage script
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the example script to see the anomaly detection in action:
```bash
python example.py
```

2. To use the anomaly detector in your own code:
```python
from anomaly_detector import AnomalyDetector
from utils.preprocessing import load_transaction_data, extract_features

# Load and preprocess your data
df = load_transaction_data('your_transactions.csv')
features = extract_features(df)

# Initialize and train the model
detector = AnomalyDetector(method='isolation_forest', contamination=0.05)
detector.fit(features)

# Make predictions
predictions = detector.predict(features)
```

## Model Details

### Isolation Forest
- Unsupervised learning algorithm
- Works by isolating observations in random trees
- Efficient for high-dimensional data
- Good for detecting point anomalies

### Autoencoder
- Neural network-based approach
- Learns to reconstruct normal patterns
- Detects anomalies based on reconstruction error
- Good for detecting complex patterns and contextual anomalies

## Data Requirements

The system expects transaction data with the following columns:
- timestamp: Transaction timestamp
- amount: Transaction amount
- transaction_type: Type of transaction (optional)

Additional features can be added by modifying the `extract_features` function in `preprocessing.py`.

## License

MIT License 