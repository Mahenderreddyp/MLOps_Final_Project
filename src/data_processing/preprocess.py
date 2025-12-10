"""
Main Data Preprocessing Pipeline

This script orchestrates the complete data preprocessing pipeline:
1. Data loading
2. Data cleaning
3. Feature engineering
4. Train/test split
5. Data saving
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.data_loader import DataLoader
from data_processing.clean_data import clean_sparcs_data
from data_processing.feature_engineering import engineer_sparcs_features
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Main preprocessing pipeline class"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.loader = DataLoader()
        self.logger = logger
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, source: str = 'api', **kwargs) -> pd.DataFrame:
        """
        Load data from specified source
        
        Args:
            source: Data source ('api', 'local', 'azure')
            **kwargs: Additional arguments for data loading
            
        Returns:
            Raw DataFrame
        """
        self.logger.info(f"Loading data from {source}...")
        
        if source == 'api':
            limit = kwargs.get('limit', 10000)
            df = self.loader.load_from_ny_health_api(limit=limit)
        
        elif source == 'local':
            file_path = kwargs.get('file_path', '../data/raw/sparcs_data.csv')
            df = self.loader.load_from_local(file_path)
        
        elif source == 'azure':
            datastore_name = kwargs.get('datastore_name', 'workspaceblobstore')
            file_path = kwargs.get('file_path', 'sparcs/data.csv')
            workspace = kwargs.get('workspace')
            
            self.loader.workspace = workspace
            df = self.loader.load_from_azure_datastore(datastore_name, file_path)
        
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")
        return clean_sparcs_data(df)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features...")
        return engineer_sparcs_features(df)
    
    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'log_total_costs',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple:
        """
        Split data into train and test sets
        
        Args:
            df: DataFrame with features
            target_column: Name of target variable
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info(f"Splitting data (test_size={test_size})...")
        
        # Separate features and target
        X = df.drop(columns=[target_column, 'total_costs'], errors='ignore')
        y = df[target_column]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        
        self.logger.info(f"Train set: {len(X_train)} rows")
        self.logger.info(f"Test set: {len(X_test)} rows")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str = '../data/processed'
    ) -> None:
        """
        Save processed data to files
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            output_dir: Output directory
        """
        self.logger.info(f"Saving processed data to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save train data
        train_df = X_train.copy()
        train_df['target'] = y_train
        train_df.to_csv(f'{output_dir}/train.csv', index=False)
        
        # Save test data
        test_df = X_test.copy()
        test_df['target'] = y_test
        test_df.to_csv(f'{output_dir}/test.csv', index=False)
        
        # Save feature names
        with open(f'{output_dir}/feature_names.txt', 'w') as f:
            f.write('\n'.join(X_train.columns.tolist()))
        
        self.logger.info("Data saved successfully!")
    
    def run(
        self,
        source: str = 'api',
        output_dir: str = '../data/processed',
        **kwargs
    ) -> None:
        """
        Run the complete preprocessing pipeline
        
        Args:
            source: Data source ('api', 'local', 'azure')
            output_dir: Output directory for processed data
            **kwargs: Additional arguments
        """
        self.logger.info("="*50)
        self.logger.info("STARTING PREPROCESSING PIPELINE")
        self.logger.info("="*50)
        
        # Load data
        df_raw = self.load_data(source=source, **kwargs)
        
        # Clean data
        df_clean = self.clean_data(df_raw)
        
        # Engineer features
        df_engineered = self.engineer_features(df_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df_engineered,
            target_column=kwargs.get('target_column', 'log_total_costs'),
            test_size=kwargs.get('test_size', 0.2),
            random_state=kwargs.get('random_state', 42)
        )
        
        # Save data
        self.save_processed_data(X_train, X_test, y_train, y_test, output_dir)
        
        self.logger.info("="*50)
        self.logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info("="*50)
        
        # Print summary
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        print(f"Raw data: {len(df_raw)} rows, {len(df_raw.columns)} columns")
        print(f"Cleaned data: {len(df_clean)} rows, {len(df_clean.columns)} columns")
        print(f"Engineered data: {len(df_engineered)} rows, {len(df_engineered.columns)} columns")
        print(f"Train set: {len(X_train)} rows")
        print(f"Test set: {len(X_test)} rows")
        print(f"Number of features: {len(X_train.columns)}")
        print(f"Output directory: {output_dir}")
        print("="*50 + "\n")


def main():
    """Main function to run preprocessing pipeline"""
    
    parser = argparse.ArgumentParser(description='Hospital Cost Prediction - Data Preprocessing')
    
    parser.add_argument(
        '--source',
        type=str,
        default='api',
        choices=['api', 'local', 'azure'],
        help='Data source'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Number of records to load from API'
    )
    
    parser.add_argument(
        '--file-path',
        type=str,
        help='Path to local file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (proportion)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(config_path=args.config)
    
    # Run pipeline
    kwargs = {
        'limit': args.limit,
        'file_path': args.file_path,
        'test_size': args.test_size,
        'random_state': args.random_state
    }
    
    pipeline.run(
        source=args.source,
        output_dir=args.output_dir,
        **kwargs
    )


if __name__ == "__main__":
    main()
