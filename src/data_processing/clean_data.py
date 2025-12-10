"""
Data Cleaning Module for Hospital Cost Prediction

This module handles data cleaning operations including:
- Missing value imputation
- Outlier detection and removal
- Data type conversions
- Duplicate removal
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Class to handle data cleaning operations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataCleaner
        
        Args:
            df: Input DataFrame to clean
        """
        self.df = df.copy()
        self.logger = logger
        self.cleaning_report = {
            'initial_rows': len(df),
            'initial_columns': len(df.columns),
            'operations': []
        }
    
    def handle_missing_values(
        self,
        strategy: str = 'mean',
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        drop_threshold: float = 0.5
    ) -> 'DataCleaner':
        """
        Handle missing values in the dataset
        
        Args:
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
            numerical_columns: List of numerical columns to impute
            categorical_columns: List of categorical columns to impute
            drop_threshold: Drop columns with missing % above this threshold
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Handling missing values...")
        
        initial_missing = self.df.isnull().sum().sum()
        
        # Drop columns with too many missing values
        missing_pct = self.df.isnull().sum() / len(self.df)
        cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()
        
        if cols_to_drop:
            self.logger.info(f"Dropping columns with >{drop_threshold*100}% missing: {cols_to_drop}")
            self.df = self.df.drop(columns=cols_to_drop)
        
        # Auto-detect column types if not provided
        if numerical_columns is None:
            numerical_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_columns is None:
            categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle numerical columns
        if numerical_columns and strategy in ['mean', 'median']:
            imputer = SimpleImputer(strategy=strategy)
            self.df[numerical_columns] = imputer.fit_transform(self.df[numerical_columns])
            self.logger.info(f"Imputed {len(numerical_columns)} numerical columns with {strategy}")
        
        # Handle categorical columns
        if categorical_columns:
            if strategy == 'mode':
                for col in categorical_columns:
                    if self.df[col].isnull().any():
                        mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                        self.df[col].fillna(mode_value, inplace=True)
            else:
                self.df[categorical_columns] = self.df[categorical_columns].fillna('Unknown')
            
            self.logger.info(f"Imputed {len(categorical_columns)} categorical columns")
        
        final_missing = self.df.isnull().sum().sum()
        
        self.cleaning_report['operations'].append({
            'operation': 'handle_missing_values',
            'initial_missing': initial_missing,
            'final_missing': final_missing,
            'columns_dropped': cols_to_drop
        })
        
        return self
    
    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Remove duplicate rows
        
        Args:
            subset: Columns to consider for identifying duplicates
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Removing duplicates...")
        
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep='first')
        final_rows = len(self.df)
        
        duplicates_removed = initial_rows - final_rows
        
        self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        self.cleaning_report['operations'].append({
            'operation': 'remove_duplicates',
            'duplicates_removed': duplicates_removed
        })
        
        return self
    
    def convert_data_types(
        self,
        column_types: Dict[str, str]
    ) -> 'DataCleaner':
        """
        Convert column data types
        
        Args:
            column_types: Dictionary mapping column names to desired types
                         e.g., {'age': 'int', 'cost': 'float'}
        
        Returns:
            Self for method chaining
        """
        self.logger.info("Converting data types...")
        
        for col, dtype in column_types.items():
            if col in self.df.columns:
                try:
                    if dtype == 'numeric':
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    elif dtype == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    elif dtype == 'category':
                        self.df[col] = self.df[col].astype('category')
                    else:
                        self.df[col] = self.df[col].astype(dtype)
                    
                    self.logger.info(f"Converted {col} to {dtype}")
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
        
        self.cleaning_report['operations'].append({
            'operation': 'convert_data_types',
            'columns_converted': list(column_types.keys())
        })
        
        return self
    
    def remove_outliers(
        self,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> 'DataCleaner':
        """
        Remove outliers from specified columns
        
        Args:
            columns: List of columns to check for outliers
            method: Method to use ('iqr', 'zscore')
            threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Removing outliers using {method} method...")
        
        initial_rows = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                self.df = self.df[
                    (self.df[col] >= lower_bound) & 
                    (self.df[col] <= upper_bound)
                ]
                
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
        
        final_rows = len(self.df)
        outliers_removed = initial_rows - final_rows
        
        self.logger.info(f"Removed {outliers_removed} outlier rows")
        
        self.cleaning_report['operations'].append({
            'operation': 'remove_outliers',
            'method': method,
            'columns': columns,
            'outliers_removed': outliers_removed
        })
        
        return self
    
    def cap_outliers(
        self,
        columns: List[str],
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
    ) -> 'DataCleaner':
        """
        Cap outliers at specified percentiles instead of removing
        
        Args:
            columns: List of columns to cap
            lower_percentile: Lower percentile for capping
            upper_percentile: Upper percentile for capping
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Capping outliers at {lower_percentile} and {upper_percentile} percentiles...")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            lower_cap = self.df[col].quantile(lower_percentile)
            upper_cap = self.df[col].quantile(upper_percentile)
            
            self.df[col] = self.df[col].clip(lower=lower_cap, upper=upper_cap)
            self.logger.info(f"Capped {col}: [{lower_cap:.2f}, {upper_cap:.2f}]")
        
        self.cleaning_report['operations'].append({
            'operation': 'cap_outliers',
            'columns': columns,
            'lower_percentile': lower_percentile,
            'upper_percentile': upper_percentile
        })
        
        return self
    
    def standardize_text(
        self,
        columns: List[str],
        lowercase: bool = True,
        strip: bool = True,
        replace_spaces: bool = False
    ) -> 'DataCleaner':
        """
        Standardize text in specified columns
        
        Args:
            columns: List of text columns to standardize
            lowercase: Convert to lowercase
            strip: Strip leading/trailing whitespace
            replace_spaces: Replace spaces with underscores
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Standardizing text columns...")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if lowercase:
                self.df[col] = self.df[col].str.lower()
            
            if strip:
                self.df[col] = self.df[col].str.strip()
            
            if replace_spaces:
                self.df[col] = self.df[col].str.replace(' ', '_')
            
            self.logger.info(f"Standardized text in {col}")
        
        self.cleaning_report['operations'].append({
            'operation': 'standardize_text',
            'columns': columns
        })
        
        return self
    
    def filter_rows(
        self,
        conditions: Dict[str, any]
    ) -> 'DataCleaner':
        """
        Filter rows based on conditions
        
        Args:
            conditions: Dictionary of column: condition pairs
                       e.g., {'age': lambda x: x > 0, 'cost': lambda x: x < 1000000}
        
        Returns:
            Self for method chaining
        """
        self.logger.info("Filtering rows based on conditions...")
        
        initial_rows = len(self.df)
        
        for col, condition in conditions.items():
            if col in self.df.columns:
                if callable(condition):
                    self.df = self.df[condition(self.df[col])]
                else:
                    self.df = self.df[self.df[col] == condition]
        
        final_rows = len(self.df)
        rows_filtered = initial_rows - final_rows
        
        self.logger.info(f"Filtered out {rows_filtered} rows")
        
        self.cleaning_report['operations'].append({
            'operation': 'filter_rows',
            'rows_filtered': rows_filtered
        })
        
        return self
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        self.cleaning_report['final_rows'] = len(self.df)
        self.cleaning_report['final_columns'] = len(self.df.columns)
        
        return self.df
    
    def get_report(self) -> Dict:
        """
        Get cleaning report
        
        Returns:
            Dictionary containing cleaning statistics
        """
        return self.cleaning_report
    
    def print_report(self) -> None:
        """Print cleaning report"""
        print("\n" + "="*50)
        print("DATA CLEANING REPORT")
        print("="*50)
        print(f"Initial shape: ({self.cleaning_report['initial_rows']}, {self.cleaning_report['initial_columns']})")
        print(f"Final shape: ({self.cleaning_report['final_rows']}, {self.cleaning_report['final_columns']})")
        print(f"\nRows removed: {self.cleaning_report['initial_rows'] - self.cleaning_report['final_rows']}")
        print(f"Columns removed: {self.cleaning_report['initial_columns'] - self.cleaning_report['final_columns']}")
        print(f"\nOperations performed: {len(self.cleaning_report['operations'])}")
        for i, op in enumerate(self.cleaning_report['operations'], 1):
            print(f"\n{i}. {op['operation']}")
            for key, value in op.items():
                if key != 'operation':
                    print(f"   - {key}: {value}")
        print("="*50 + "\n")


def clean_sparcs_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to clean SPARCS hospital data
    
    Args:
        df: Raw SPARCS DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting SPARCS data cleaning pipeline...")
    
    # Initialize cleaner
    cleaner = DataCleaner(df)
    
    # Define data type conversions
    type_conversions = {
        'total_costs': 'numeric',
        'total_charges': 'numeric',
        'length_of_stay': 'numeric'
    }
    
    # Clean data
    cleaned_df = (cleaner
        .remove_duplicates()
        .convert_data_types(type_conversions)
        .filter_rows({
            'total_costs': lambda x: (x > 0) & (x < 10000000),  # Remove invalid costs
            'length_of_stay': lambda x: x >= 0
        })
        .handle_missing_values(
            strategy='median',
            drop_threshold=0.7
        )
        .get_cleaned_data()
    )
    
    # Print report
    cleaner.print_report()
    
    logger.info("Data cleaning completed successfully!")
    
    return cleaned_df


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_from_ny_health_api(limit=5000)
    
    # Clean data
    cleaned_df = clean_sparcs_data(df)
    
    print(f"\nCleaned data shape: {cleaned_df.shape}")
    print(f"\nFirst few rows:")
    print(cleaned_df.head())
