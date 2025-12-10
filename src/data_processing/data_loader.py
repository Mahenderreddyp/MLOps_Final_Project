"""
Data Loader Module for Hospital Cost Prediction

This module handles loading data from various sources including:
- Azure ML Datastores
- Local files
- NY Health Data API
"""

import os
import pandas as pd
import logging
from typing import Optional, Dict, Any, Union
import requests
from azureml.core import Workspace, Dataset, Datastore
from azureml.data.datapath import DataPath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class to handle data loading from multiple sources"""
    
    def __init__(self, workspace: Optional[Workspace] = None):
        """
        Initialize DataLoader
        
        Args:
            workspace: Azure ML Workspace object (optional)
        """
        self.workspace = workspace
        self.logger = logger
    
    def load_from_azure_datastore(
        self, 
        datastore_name: str, 
        file_path: str,
        dataset_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from Azure ML Datastore
        
        Args:
            datastore_name: Name of the datastore
            file_path: Path to file in datastore
            dataset_name: Optional name to register dataset
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            ValueError: If workspace is not provided
            Exception: If loading fails
        """
        if not self.workspace:
            raise ValueError("Workspace must be provided to load from Azure")
        
        self.logger.info(f"Loading data from datastore: {datastore_name}")
        
        try:
            # Get datastore
            datastore = Datastore.get(self.workspace, datastore_name)
            
            # Create dataset
            dataset = Dataset.Tabular.from_delimited_files(
                path=DataPath(datastore, file_path)
            )
            
            # Register dataset if name provided
            if dataset_name:
                dataset = dataset.register(
                    workspace=self.workspace,
                    name=dataset_name,
                    description=f"Hospital discharge data from {file_path}",
                    create_new_version=True
                )
                self.logger.info(f"Dataset registered as: {dataset_name}")
            
            # Convert to pandas DataFrame
            df = dataset.to_pandas_dataframe()
            self.logger.info(f"Loaded {len(df)} rows from Azure datastore")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading from Azure: {str(e)}")
            raise
    
    def load_from_local(
        self, 
        file_path: str, 
        file_type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from local file
        
        Args:
            file_path: Path to local file
            file_type: Type of file (csv, parquet, excel). Auto-detected if None
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported
        """
        self.logger.info(f"Loading data from local file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Auto-detect file type if not provided
            if file_type is None:
                file_type = os.path.splitext(file_path)[1].lower()
            
            # Load based on file type
            if file_type in ['.csv', 'csv']:
                df = pd.read_csv(file_path, **kwargs)
            elif file_type in ['.parquet', 'parquet']:
                df = pd.read_parquet(file_path, **kwargs)
            elif file_type in ['.xlsx', '.xls', 'excel']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_type in ['.json', 'json']:
                df = pd.read_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from local file")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading from local file: {str(e)}")
            raise
    
    def load_from_ny_health_api(
        self, 
        limit: Optional[int] = 10000,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> pd.DataFrame:
        """
        Load data from NY Health Data API (SPARCS)
        
        Args:
            limit: Maximum number of records to fetch
            offset: Starting record position
            filters: Dictionary of filters to apply
            timeout: Request timeout in seconds
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            Exception: If API request fails
        """
        base_url = "https://health.data.ny.gov/resource/sf4k-39ay.json"
        
        self.logger.info(f"Loading data from NY Health Data API (limit={limit}, offset={offset})")
        
        try:
            # Build query parameters
            params = {
                "$limit": limit,
                "$offset": offset,
                "$order": ":id"
            }
            
            # Add filters if provided
            if filters:
                for key, value in filters.items():
                    params[key] = value
            
            # Make API request
            response = requests.get(base_url, params=params, timeout=timeout)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            df = pd.DataFrame(data)
            
            self.logger.info(f"Successfully loaded {len(df)} rows from API")
            return df
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error loading from API: {str(e)}")
            raise
    
    def load_from_registered_dataset(
        self,
        dataset_name: str,
        version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from Azure ML registered dataset
        
        Args:
            dataset_name: Name of registered dataset
            version: Specific version to load (latest if None)
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            ValueError: If workspace is not provided
        """
        if not self.workspace:
            raise ValueError("Workspace must be provided to load registered dataset")
        
        self.logger.info(f"Loading registered dataset: {dataset_name}")
        
        try:
            if version:
                dataset = Dataset.get_by_name(
                    workspace=self.workspace,
                    name=dataset_name,
                    version=version
                )
            else:
                dataset = Dataset.get_by_name(
                    workspace=self.workspace,
                    name=dataset_name
                )
            
            df = dataset.to_pandas_dataframe()
            self.logger.info(f"Loaded {len(df)} rows from registered dataset")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading registered dataset: {str(e)}")
            raise
    
    def save_to_local(
        self,
        df: pd.DataFrame,
        file_path: str,
        file_type: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Save DataFrame to local file
        
        Args:
            df: DataFrame to save
            file_path: Path where to save file
            file_type: Type of file (csv, parquet, excel). Auto-detected if None
            **kwargs: Additional arguments for pandas write functions
        """
        self.logger.info(f"Saving data to local file: {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Auto-detect file type if not provided
        if file_type is None:
            file_type = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_type in ['.csv', 'csv']:
                df.to_csv(file_path, index=False, **kwargs)
            elif file_type in ['.parquet', 'parquet']:
                df.to_parquet(file_path, index=False, **kwargs)
            elif file_type in ['.xlsx', 'excel']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_type in ['.json', 'json']:
                df.to_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"Successfully saved {len(df)} rows to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving to file: {str(e)}")
            raise


def main():
    """Example usage of DataLoader"""
    
    # Initialize loader
    loader = DataLoader()
    
    # Example 1: Load from NY Health API
    print("Loading sample data from NY Health API...")
    df = loader.load_from_ny_health_api(limit=1000)
    print(f"Loaded {len(df)} rows")
    print(df.head())
    
    # Example 2: Save to local
    print("\nSaving to local CSV...")
    loader.save_to_local(df, "../data/raw/sample_data.csv")
    
    # Example 3: Load from local
    print("\nLoading from local CSV...")
    df_local = loader.load_from_local("../data/raw/sample_data.csv")
    print(f"Loaded {len(df_local)} rows")


if __name__ == "__main__":
    main()
