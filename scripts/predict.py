import pandas as pd
import numpy as np
import argparse
import joblib
import os
from tensorflow.keras.models import load_model

def load_resources(model_dir):
    # Load trained model
    model = load_model(os.path.join(model_dir, 'churn_model.h5'))

    # Load scaler
    scaler = joblib.load(os.path.join(model_dir, 'scaler.save'))

    # Load feature list
    features = joblib.load(os.path.join(model_dir, 'train_features.save'))

    return model, scaler, features

def preprocess_input(data, features, scaler):
    # One-hot encode categorical columns (similar to training)
    data_encoded = pd.get_dummies(data)

    # Add any missing columns from features list, fill with 0
    for col in features:
        if col not in data_encoded.columns:
            data_encoded[col] = 0

    # Keep only columns in the same order as features
    data_encoded = data_encoded[features]

    # Scale the data
    data_scaled = scaler.transform(data_encoded)

    return data_scaled


def predict_churn(model, data_scaled):
    preds = model.predict(data_scaled)
    preds_binary = (preds > 0.5).astype(int).flatten()
    return preds_binary, preds.flatten()

def main():
    parser = argparse.ArgumentParser(description='Predict customer churn.')
    parser.add_argument('--input', type=str, default='data/new_customers.csv', help='Path to input CSV')
    parser.add_argument('--output', type=str, default='data/predictions.csv', help='Path to save predictions')
    parser.add_argument('--model_dir', type=str, default='models', help='Path to model directory')
    args = parser.parse_args()

    print(f"Loading input file: {args.input}")
    df = pd.read_csv(args.input)

    print("Loading model and scaler...")
    model, scaler, features = load_resources(args.model_dir)

    print("Preprocessing input data...")
    X_scaled = preprocess_input(df, features, scaler)

    print("Running predictions...")
    binary_preds, prob_preds = predict_churn(model, X_scaled)

    print("Saving predictions...")
    df['churn_prediction'] = binary_preds
    df['churn_probability'] = prob_preds
    df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
