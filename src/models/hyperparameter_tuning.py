"""
Hyperparameter Tuning Module for Hospital Cost Prediction

This module handles hyperparameter optimization including:
- Grid search
- Random search
- Bayesian optimization
- Azure ML Hyperdrive integration
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    KFold
)
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Class for hyperparameter tuning"""
    
    def __init__(
        self,
        base_model: Optional[Any] = None,
        cv_folds: int = 5,
        scoring: str = 'r2',
        random_state: int = 42
    ):
        """
        Initialize HyperparameterTuner
        
        Args:
            base_model: Base model class or instance
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            random_state: Random seed
        """
        self.base_model = base_model or GradientBoostingRegressor
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.logger = logger
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.tuning_history = []
    
    def get_default_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get default parameter grid for Boosted Decision Tree
        
        Returns:
            Dictionary of parameter grids
        """
        return {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
    
    def get_default_param_distribution(self) -> Dict[str, Any]:
        """
        Get default parameter distribution for random search
        
        Returns:
            Dictionary of parameter distributions
        """
        return {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'subsample': uniform(0.6, 0.4),  # 0.6 to 1.0
            'max_features': ['sqrt', 'log2', None]
        }
    
    def grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_jobs: int = -1,
        verbose: int = 2
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid (uses default if None)
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        self.logger.info("Starting Grid Search hyperparameter tuning...")
        
        if param_grid is None:
            param_grid = self.get_default_param_grid()
        
        # Initialize base model
        base_model = self.base_model(random_state=self.random_state)
        
        # Create scorer
        scorer = make_scorer(r2_score)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        self.logger.info(f"Fitting grid search with {len(param_grid)} parameter combinations...")
        grid_search.fit(X, y)
        
        # Store results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # Store tuning history
        self.tuning_history.append({
            'method': 'grid_search',
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'cv_scores': {
                'mean': float(grid_search.cv_results_['mean_test_score'].max()),
                'std': float(grid_search.cv_results_['std_test_score'][
                    grid_search.cv_results_['mean_test_score'].argmax()
                ])
            }
        })
        
        self.logger.info(f"Grid Search completed!")
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {self.best_score:.4f}")
        
        return self.best_model, self.best_params, self.best_score
    
    def random_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_distribution: Optional[Dict[str, Any]] = None,
        n_iter: int = 50,
        n_jobs: int = -1,
        verbose: int = 2
    ) -> Tuple[Any, Dict[str, Any], float]:
        """
        Perform random search for hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target variable
            param_distribution: Parameter distribution (uses default if None)
            n_iter: Number of iterations
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        self.logger.info(f"Starting Random Search hyperparameter tuning (n_iter={n_iter})...")
        
        if param_distribution is None:
            param_distribution = self.get_default_param_distribution()
        
        # Initialize base model
        base_model = self.base_model(random_state=self.random_state)
        
        # Create scorer
        scorer = make_scorer(r2_score)
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distribution,
            n_iter=n_iter,
            cv=self.cv_folds,
            scoring=scorer,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=self.random_state,
            return_train_score=True
        )
        
        self.logger.info(f"Fitting random search with {n_iter} iterations...")
        random_search.fit(X, y)
        
        # Store results
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        # Store tuning history
        self.tuning_history.append({
            'method': 'random_search',
            'n_iter': n_iter,
            'best_params': self.best_params,
            'best_score': float(self.best_score),
            'cv_scores': {
                'mean': float(random_search.cv_results_['mean_test_score'].max()),
                'std': float(random_search.cv_results_['std_test_score'][
                    random_search.cv_results_['mean_test_score'].argmax()
                ])
            }
        })
        
        self.logger.info(f"Random Search completed!")
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {self.best_score:.4f}")
        
        return self.best_model, self.best_params, self.best_score
    
    def evaluate_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate best model on train and test sets
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run tuning first.")
        
        self.logger.info("Evaluating best model...")
        
        # Train predictions
        train_pred = self.best_model.predict(X_train)
        train_r2 = r2_score(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        # Test predictions
        test_pred = self.best_model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        metrics = {
            'train_r2': float(train_r2),
            'train_rmse': float(train_rmse),
            'test_r2': float(test_r2),
            'test_rmse': float(test_rmse)
        }
        
        self.logger.info(f"Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}")
        self.logger.info(f"Test R²: {test_r2:.4f}, Test RMSE: {test_rmse:.4f}")
        
        return metrics
    
    def save_best_model(
        self,
        model_path: str,
        include_metadata: bool = True
    ) -> None:
        """
        Save best tuned model
        
        Args:
            model_path: Path to save model
            include_metadata: Whether to save metadata
        """
        if self.best_model is None:
            raise ValueError("No best model found. Run tuning first.")
        
        self.logger.info(f"Saving best model to {model_path}...")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, model_path)
        
        # Save metadata
        if include_metadata:
            metadata_path = model_path.replace('.pkl', '_tuning_metadata.json')
            metadata = {
                'best_params': self.best_params,
                'best_score': float(self.best_score) if self.best_score else None,
                'tuning_history': self.tuning_history,
                'cv_folds': self.cv_folds,
                'scoring': self.scoring
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Metadata saved to {metadata_path}")
        
        self.logger.info("Model saved successfully!")
    
    def print_summary(self) -> None:
        """Print tuning summary"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*60)
        
        if self.best_params:
            print("\nBest Parameters:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
        
        if self.best_score:
            print(f"\nBest CV Score: {self.best_score:.4f}")
        
        if self.tuning_history:
            print(f"\nTuning Methods Used: {len(self.tuning_history)}")
            for i, history in enumerate(self.tuning_history, 1):
                print(f"\n{i}. {history['method']}")
                print(f"   Best Score: {history['best_score']:.4f}")
        
        print("="*60 + "\n")


def create_azure_hyperdrive_config(
    param_sampling: str = 'random',
    param_space: Optional[Dict] = None
) -> Dict:
    """
    Create Azure ML Hyperdrive configuration
    
    Args:
        param_sampling: Sampling method ('random', 'grid', 'bayesian')
        param_space: Parameter space dictionary
        
    Returns:
        Hyperdrive configuration dictionary
        
    Note:
        This function returns a configuration dict. To use with Azure ML Hyperdrive,
        import: from azureml.train.hyperdrive import choice, uniform, RandomParameterSampling
        and use the param_space values directly in HyperDriveConfig.
    """
    # Note: For actual Azure ML usage, import:
    # from azureml.train.hyperdrive import choice, uniform, RandomParameterSampling, GridParameterSampling
    
    if param_space is None:
        # These would be used with Azure ML Hyperdrive choice() and uniform() functions
        # Example usage:
        # from azureml.train.hyperdrive import choice, uniform
        # param_space = {
        #     '--n_estimators': choice(100, 200, 300),
        #     '--learning_rate': uniform(0.01, 0.3),
        #     '--max_depth': choice(3, 5, 7, 10),
        #     '--min_samples_split': choice(2, 5, 10),
        #     '--min_samples_leaf': choice(1, 2, 4)
        # }
        param_space = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    config = {
        'run_config': None,  # Will be set with RunConfiguration
        'hyperparameter_sampling': param_sampling,
        'primary_metric_name': 'r2_score',
        'primary_metric_goal': 'maximize',
        'max_total_runs': 50,
        'max_concurrent_runs': 4,
        'policy': None  # Will be set with BanditPolicy or MedianStoppingPolicy
    }
    
    return config


def main():
    """Main function to perform hyperparameter tuning"""
    
    parser = argparse.ArgumentParser(description='Hospital Cost Prediction - Hyperparameter Tuning')
    
    parser.add_argument(
        '--train-data',
        type=str,
        required=True,
        help='Path to training data (CSV)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        help='Path to test data (CSV) for evaluation'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='target',
        help='Name of target column'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='random',
        choices=['grid', 'random'],
        help='Tuning method'
    )
    
    parser.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='Number of iterations for random search'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--output-model',
        type=str,
        default='../models/best_tuned_model.pkl',
        help='Path to save best model'
    )
    
    args = parser.parse_args()
    
    # Load training data
    logger.info(f"Loading training data from {args.train_data}...")
    train_df = pd.read_csv(args.train_data)
    
    # Separate features and target
    X_train = train_df.drop(columns=[args.target_column])
    y_train = train_df[args.target_column]
    
    # Select numeric features only
    X_train = X_train.select_dtypes(include=[np.number])
    
    logger.info(f"Training set shape: {X_train.shape}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        cv_folds=args.cv_folds,
        random_state=42
    )
    
    # Perform tuning
    if args.method == 'grid':
        best_model, best_params, best_score = tuner.grid_search(X_train, y_train)
    else:
        best_model, best_params, best_score = tuner.random_search(
            X_train,
            y_train,
            n_iter=args.n_iter
        )
    
    # Evaluate on test set if provided
    if args.test_data:
        logger.info("Evaluating on test set...")
        test_df = pd.read_csv(args.test_data)
        X_test = test_df.drop(columns=[args.target_column])
        y_test = test_df[args.target_column]
        X_test = X_test.select_dtypes(include=[np.number])
        
        metrics = tuner.evaluate_best_model(X_train, y_train, X_test, y_test)
    
    # Save best model
    tuner.save_best_model(args.output_model)
    
    # Print summary
    tuner.print_summary()
    
    logger.info(f"Hyperparameter tuning completed! Best model saved to {args.output_model}")


if __name__ == "__main__":
    main()

