"""
Model Training Module for Hospital Cost Prediction

This module handles model training including:
- Boosted Decision Tree Regression
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Model saving
"""

import os
import sys
import argparse
import logging
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HospitalCostPredictor:
    """Main model training class"""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize predictor
        
        Args:
            model_params: Model hyperparameters
        """
        self.model_params = model_params or self._get_default_params()
        self.model = None
        self.best_params = None
        self.training_history = {}
        self.logger = logger
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters"""
        return {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 7,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8,
            'random_state': 42,
            'verbose': 1
        }
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        validate: bool = True,
        cv_folds: int = 5
    ) -> 'HospitalCostPredictor':
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            validate: Whether to perform cross-validation
            cv_folds: Number of CV folds
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Training Boosted Decision Tree Regressor...")
        self.logger.info(f"Training set: {X_train.shape}")
        self.logger.info(f"Model parameters: {self.model_params}")
        
        # Initialize model
        self.model = GradientBoostingRegressor(**self.model_params)
        
        # Cross-validation
        if validate:
            self.logger.info(f"Performing {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            
            self.training_history['cv_scores'] = cv_scores.tolist()
            self.training_history['cv_mean'] = float(cv_scores.mean())
            self.training_history['cv_std'] = float(cv_scores.std())
            
            self.logger.info(f"CV R² scores: {cv_scores}")
            self.logger.info(f"CV mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train on full training set
        self.logger.info("Training on full training set...")
        self.model.fit(X_train, y_train)
        
        # Get training score
        train_pred = self.model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        
        self.training_history['train_r2'] = float(train_r2)
        self.logger.info(f"Training R²: {train_r2:.4f}")
        
        return self
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, list]] = None,
        search_method: str = 'grid',
        cv_folds: int = 5,
        n_iter: int = 50
    ) -> 'HospitalCostPredictor':
        """
        Tune hyperparameters using grid search or random search
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for search
            search_method: 'grid' or 'random'
            cv_folds: Number of CV folds
            n_iter: Number of iterations for random search
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Tuning hyperparameters using {search_method} search...")
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        # Initialize base model
        base_model = GradientBoostingRegressor(random_state=42)
        
        # Perform search
        if search_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=2
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=2
            )
        
        # Fit search
        search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        
        self.training_history['best_params'] = self.best_params
        self.training_history['best_cv_score'] = float(search.best_score_)
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return self
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model on test set...")
        
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'r2_score': float(r2_score(y_test, y_pred)),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred) * 100)
        }
        
        # Store metrics
        self.training_history['test_metrics'] = metrics
        
        # Log metrics
        self.logger.info("Test Set Metrics:")
        self.logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        self.logger.info(f"  MAE: {metrics['mae']:.4f}")
        self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        self.logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def get_feature_importance(
        self,
        feature_names: list,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        self.logger.info(f"\nTop {top_n} Feature Importances:")
        for idx, row in importance_df.iterrows():
            self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_model(
        self,
        model_path: str,
        include_metadata: bool = True
    ) -> None:
        """
        Save trained model
        
        Args:
            model_path: Path to save model
            include_metadata: Whether to save metadata
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.logger.info(f"Saving model to {model_path}...")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        if include_metadata:
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            self.logger.info(f"Metadata saved to {metadata_path}")
        
        self.logger.info("Model saved successfully!")
    
    def load_model(self, model_path: str) -> 'HospitalCostPredictor':
        """
        Load trained model
        
        Args:
            model_path: Path to model file
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Loading model from {model_path}...")
        
        self.model = joblib.load(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.training_history = json.load(f)
        
        self.logger.info("Model loaded successfully!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)


def main():
    """Main function to train model"""
    
    parser = argparse.ArgumentParser(description='Hospital Cost Prediction - Model Training')
    
    parser.add_argument(
        '--train-data',
        type=str,
        default='../data/processed/train.csv',
        help='Path to training data'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        default='../data/processed/test.csv',
        help='Path to test data'
    )
    
    parser.add_argument(
        '--output-model',
        type=str,
        default='../models/boosted_tree_model.pkl',
        help='Path to save trained model'
    )
    
    parser.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model configuration file (YAML)'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        import yaml
        logger.info(f"Loading configuration from {args.config}...")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update model params from config if available
        if 'model' in config and 'hyperparameters' in config['model']:
            # This would be used to override default params
            logger.info("Configuration loaded successfully")
    
    # Load data
    logger.info("Loading training and test data...")
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    
    # Separate features and target
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    # Select only numeric columns
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Initialize predictor
    predictor = HospitalCostPredictor()
    
    # Train or tune
    if args.tune_hyperparameters:
        predictor.tune_hyperparameters(
            X_train,
            y_train,
            cv_folds=args.cv_folds
        )
    else:
        predictor.train(
            X_train,
            y_train,
            cv_folds=args.cv_folds
        )
    
    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)
    
    # Feature importance
    importance_df = predictor.get_feature_importance(X_train.columns.tolist())
    
    # Save model
    predictor.save_model(args.output_model)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Model saved to: {args.output_model}")
    print(f"Test R² Score: {metrics['r2_score']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
