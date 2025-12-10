"""
Model Prediction Module

This script handles making predictions with trained model
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load trained model"""
    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully!")
    return model


def make_predictions(
    model,
    X: pd.DataFrame,
    inverse_transform: bool = True
) -> np.ndarray:
    """
    Make predictions using trained model
    
    Args:
        model: Trained model
        X: Features
        inverse_transform: Whether to inverse log transform
        
    Returns:
        Predictions array
    """
    logger.info(f"Making predictions for {len(X)} samples...")
    
    # Make predictions
    predictions = model.predict(X)
    
    # Inverse log transform if needed
    if inverse_transform:
        predictions = np.expm1(predictions)  # exp(x) - 1
    
    return predictions


def main():
    """Main prediction function"""
    
    parser = argparse.ArgumentParser(description='Hospital Cost Prediction - Make Predictions')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data (CSV)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions'
    )
    
    parser.add_argument(
        '--inverse-transform',
        action='store_true',
        help='Inverse log transform predictions'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Load input data
    logger.info(f"Loading input data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Select numeric features only
    X = df.select_dtypes(include=[np.number])
    
    logger.info(f"Input data shape: {X.shape}")
    
    # Make predictions
    predictions = make_predictions(model, X, args.inverse_transform)
    
    # Create output dataframe
    output_df = df.copy()
    output_df['predicted_cost'] = predictions
    
    # Save predictions
    logger.info(f"Saving predictions to {args.output}...")
    output_df.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Number of predictions: {len(predictions)}")
    print(f"Mean predicted cost: ${predictions.mean():,.2f}")
    print(f"Median predicted cost: ${np.median(predictions):,.2f}")
    print(f"Min predicted cost: ${predictions.min():,.2f}")
    print(f"Max predicted cost: ${predictions.max():,.2f}")
    print(f"Predictions saved to: {args.output}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
