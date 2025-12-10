"""
Model Evaluation Module for Hospital Cost Prediction

This module handles comprehensive model evaluation including:
- Performance metrics calculation
- Visualization
- Model comparison
- Feature importance analysis
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    median_absolute_error
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize ModelEvaluator
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.logger = logger
        self.evaluation_results = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        inverse_transform: bool = False
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            inverse_transform: Whether predictions are log-transformed
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info("Calculating evaluation metrics...")
        
        # Inverse transform if needed
        if inverse_transform:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
        
        # Calculate metrics
        metrics = {
            'r2_score': float(r2_score(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mape': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
            'explained_variance': float(explained_variance_score(y_true, y_pred)),
            'median_ae': float(median_absolute_error(y_true, y_pred)),
            'mean_prediction': float(np.mean(y_pred)),
            'mean_actual': float(np.mean(y_true)),
            'std_prediction': float(np.std(y_pred)),
            'std_actual': float(np.std(y_true))
        }
        
        # Additional metrics
        metrics['mae_percentage'] = (metrics['mae'] / metrics['mean_actual']) * 100 if metrics['mean_actual'] > 0 else 0
        metrics['rmse_percentage'] = (metrics['rmse'] / metrics['mean_actual']) * 100 if metrics['mean_actual'] > 0 else 0
        
        self.evaluation_results['metrics'] = metrics
        
        # Log metrics
        self.logger.info("Evaluation Metrics:")
        self.logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        self.logger.info(f"  MAE: ${metrics['mae']:,.2f}")
        self.logger.info(f"  RMSE: ${metrics['rmse']:,.2f}")
        self.logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        self.logger.info(f"  Explained Variance: {metrics['explained_variance']:.4f}")
        self.logger.info(f"  Median AE: ${metrics['median_ae']:,.2f}")
        
        return metrics
    
    def calculate_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        inverse_transform: bool = False
    ) -> np.ndarray:
        """
        Calculate prediction residuals
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            inverse_transform: Whether predictions are log-transformed
            
        Returns:
            Array of residuals
        """
        if inverse_transform:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
        
        residuals = y_true - y_pred
        return residuals
    
    def calculate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
        random_state: int = 42,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Calculate permutation feature importance
        
        Args:
            X: Feature matrix
            y: Target variable
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        self.logger.info("Calculating permutation feature importance...")
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='r2',
            n_jobs=-1
        )
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names[:len(perm_importance.importances_mean)],
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False).head(top_n)
        
        self.evaluation_results['feature_importance'] = importance_df.to_dict('records')
        
        self.logger.info(f"\nTop {top_n} Feature Importances (Permutation):")
        for idx, row in importance_df.iterrows():
            self.logger.info(
                f"  {row['feature']}: {row['importance_mean']:.4f} "
                f"(±{row['importance_std']:.4f})"
            )
        
        return importance_df
    
    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        inverse_transform: bool = False
    ) -> None:
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save plot
            inverse_transform: Whether predictions are log-transformed
        """
        self.logger.info("Creating predictions vs actual plot...")
        
        if inverse_transform:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels
        plt.xlabel('Actual Cost ($)', fontsize=12)
        plt.ylabel('Predicted Cost ($)', fontsize=12)
        plt.title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        plt.close()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        inverse_transform: bool = False
    ) -> None:
        """
        Plot residual analysis
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save plot
            inverse_transform: Whether predictions are log-transformed
        """
        self.logger.info("Creating residual plots...")
        
        if inverse_transform:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
        
        residuals = self.calculate_residuals(y_true, y_pred, inverse_transform=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 0].set_xlabel('Predicted Cost ($)')
        axes[0, 0].set_ylabel('Residuals ($)')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Index
        axes[1, 1].plot(residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Residuals ($)')
        axes[1, 1].set_title('Residuals Over Samples')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Residual plots saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        save_path: Optional[str] = None,
        top_n: int = 20
    ) -> None:
        """
        Plot feature importance
        
        Args:
            importance_df: DataFrame with feature importances
            save_path: Path to save plot
            top_n: Number of top features to plot
        """
        self.logger.info("Creating feature importance plot...")
        
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_features['importance_mean'], xerr=top_features['importance_std'])
        plt.yticks(y_pos, top_features['feature'])
        plt.xlabel('Permutation Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def save_evaluation_results(
        self,
        output_path: str,
        include_plots: bool = False
    ) -> None:
        """
        Save evaluation results to file
        
        Args:
            output_path: Path to save results
            include_plots: Whether to include plot paths in results
        """
        self.logger.info(f"Saving evaluation results to {output_path}...")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        self.logger.info("Evaluation results saved successfully!")
    
    def print_summary(self) -> None:
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        if 'metrics' in self.evaluation_results:
            metrics = self.evaluation_results['metrics']
            print("\nPerformance Metrics:")
            print(f"  R² Score: {metrics['r2_score']:.4f}")
            print(f"  MAE: ${metrics['mae']:,.2f} ({metrics['mae_percentage']:.2f}%)")
            print(f"  RMSE: ${metrics['rmse']:,.2f} ({metrics['rmse_percentage']:.2f}%)")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Explained Variance: {metrics['explained_variance']:.4f}")
            print(f"  Median AE: ${metrics['median_ae']:,.2f}")
            print(f"\n  Mean Actual: ${metrics['mean_actual']:,.2f}")
            print(f"  Mean Predicted: ${metrics['mean_prediction']:,.2f}")
        
        if 'feature_importance' in self.evaluation_results:
            print("\nTop 5 Features:")
            for i, feat in enumerate(self.evaluation_results['feature_importance'][:5], 1):
                print(f"  {i}. {feat['feature']}: {feat['importance_mean']:.4f}")
        
        print("="*60 + "\n")


def main():
    """Main function to evaluate model"""
    
    parser = argparse.ArgumentParser(description='Hospital Cost Prediction - Model Evaluation')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data (CSV)'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='target',
        help='Name of target column'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../outputs',
        help='Output directory for results and plots'
    )
    
    parser.add_argument(
        '--inverse-transform',
        action='store_true',
        help='Inverse log transform predictions'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate evaluation plots'
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    model = joblib.load(args.model)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)
    
    # Separate features and target
    X_test = test_df.drop(columns=[args.target_column])
    y_test = test_df[args.target_column]
    
    # Select numeric features only
    X_test = X_test.select_dtypes(include=[np.number])
    
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, X_test.columns.tolist())
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(
        y_test.values,
        y_pred,
        inverse_transform=args.inverse_transform
    )
    
    # Calculate feature importance
    logger.info("Calculating feature importance...")
    importance_df = evaluator.calculate_feature_importance(X_test, y_test)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots if requested
    if args.plot:
        logger.info("Generating evaluation plots...")
        
        evaluator.plot_predictions_vs_actual(
            y_test.values,
            y_pred,
            save_path=os.path.join(args.output_dir, 'predictions_vs_actual.png'),
            inverse_transform=args.inverse_transform
        )
        
        evaluator.plot_residuals(
            y_test.values,
            y_pred,
            save_path=os.path.join(args.output_dir, 'residuals_analysis.png'),
            inverse_transform=args.inverse_transform
        )
        
        evaluator.plot_feature_importance(
            importance_df,
            save_path=os.path.join(args.output_dir, 'feature_importance.png')
        )
    
    # Save results
    evaluator.save_evaluation_results(
        os.path.join(args.output_dir, 'evaluation_results.json')
    )
    
    # Save feature importance
    importance_df.to_csv(
        os.path.join(args.output_dir, 'feature_importance.csv'),
        index=False
    )
    
    # Print summary
    evaluator.print_summary()
    
    logger.info(f"Evaluation completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

