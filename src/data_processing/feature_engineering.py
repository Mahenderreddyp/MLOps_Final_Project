"""
Feature Engineering Module for Hospital Cost Prediction

This module handles feature engineering including:
- Feature creation
- Encoding categorical variables
- Feature scaling
- Feature selection
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class to handle feature engineering operations"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer
        
        Args:
            df: Input DataFrame
        """
        self.df = df.copy()
        self.logger = logger
        self.encoders = {}
        self.scalers = {}
    
    def create_age_features(self, age_column: str = 'age_group') -> 'FeatureEngineer':
        """
        Create age-related features
        
        Args:
            age_column: Name of age group column
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Creating age features...")
        
        if age_column in self.df.columns:
            # Map age groups to numerical values
            age_mapping = {
                '0-17': 1,
                '18-29': 2,
                '30-49': 3,
                '50-69': 4,
                '70 or Older': 5,
                '70+': 5
            }
            
            self.df['age_encoded'] = self.df[age_column].map(age_mapping)
            
            # Create age risk categories
            self.df['age_risk_category'] = pd.cut(
                self.df['age_encoded'],
                bins=[0, 2, 4, 6],
                labels=['Low', 'Medium', 'High']
            )
            
            self.logger.info("Age features created")
        
        return self
    
    def create_length_of_stay_features(self, los_column: str = 'length_of_stay') -> 'FeatureEngineer':
        """
        Create length of stay related features
        
        Args:
            los_column: Name of length of stay column
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Creating length of stay features...")
        
        if los_column in self.df.columns:
            # Ensure numeric
            self.df[los_column] = pd.to_numeric(self.df[los_column], errors='coerce')
            
            # Create LOS categories
            self.df['los_category'] = pd.cut(
                self.df[los_column],
                bins=[-1, 1, 3, 7, 14, np.inf],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
            )
            
            # Log transform
            self.df['los_log'] = np.log1p(self.df[los_column])
            
            # Flag for extended stay
            self.df['extended_stay_flag'] = (self.df[los_column] > 7).astype(int)
            
            self.logger.info("Length of stay features created")
        
        return self
    
    def create_cost_features(
        self,
        cost_column: str = 'total_costs',
        charges_column: str = 'total_charges'
    ) -> 'FeatureEngineer':
        """
        Create cost-related features
        
        Args:
            cost_column: Name of total costs column
            charges_column: Name of total charges column
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Creating cost features...")
        
        # Ensure numeric
        if cost_column in self.df.columns:
            self.df[cost_column] = pd.to_numeric(self.df[cost_column], errors='coerce')
            
            # Log transform target (if using as feature)
            self.df['log_total_costs'] = np.log1p(self.df[cost_column])
            
            # Cost per day if LOS available
            if 'length_of_stay' in self.df.columns:
                self.df['cost_per_day'] = self.df[cost_column] / (self.df['length_of_stay'] + 1)
        
        if charges_column in self.df.columns:
            self.df[charges_column] = pd.to_numeric(self.df[charges_column], errors='coerce')
            
            # Cost to charge ratio
            if cost_column in self.df.columns:
                self.df['cost_to_charge_ratio'] = (
                    self.df[cost_column] / (self.df[charges_column] + 1)
                )
        
        self.logger.info("Cost features created")
        return self
    
    def create_severity_features(
        self,
        severity_column: str = 'apr_severity_of_illness_description',
        mortality_column: str = 'apr_risk_of_mortality'
    ) -> 'FeatureEngineer':
        """
        Create severity and risk features
        
        Args:
            severity_column: Name of severity column
            mortality_column: Name of mortality risk column
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Creating severity features...")
        
        # Severity encoding
        if severity_column in self.df.columns:
            severity_mapping = {
                'Minor': 1,
                'Moderate': 2,
                'Major': 3,
                'Extreme': 4
            }
            self.df['severity_encoded'] = self.df[severity_column].map(severity_mapping)
        
        # Mortality risk encoding
        if mortality_column in self.df.columns:
            mortality_mapping = {
                'Minor': 1,
                'Moderate': 2,
                'Major': 3,
                'Extreme': 4
            }
            self.df['mortality_risk_encoded'] = self.df[mortality_column].map(mortality_mapping)
        
        # Combined risk score
        if 'severity_encoded' in self.df.columns and 'mortality_risk_encoded' in self.df.columns:
            self.df['combined_risk_score'] = (
                self.df['severity_encoded'] * 0.6 + 
                self.df['mortality_risk_encoded'] * 0.4
            )
        
        self.logger.info("Severity features created")
        return self
    
    def create_admission_features(
        self,
        admission_type_column: str = 'type_of_admission',
        ed_indicator_column: str = 'emergency_department_indicator'
    ) -> 'FeatureEngineer':
        """
        Create admission-related features
        
        Args:
            admission_type_column: Name of admission type column
            ed_indicator_column: Name of ED indicator column
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Creating admission features...")
        
        # Emergency admission flag
        if admission_type_column in self.df.columns:
            self.df['emergency_admission'] = (
                self.df[admission_type_column].str.contains('Emergency', case=False, na=False)
            ).astype(int)
        
        # ED indicator
        if ed_indicator_column in self.df.columns:
            self.df['ed_flag'] = (
                self.df[ed_indicator_column] == 'Y'
            ).astype(int)
        
        self.logger.info("Admission features created")
        return self
    
    def encode_categorical_features(
        self,
        columns: List[str],
        method: str = 'onehot',
        max_categories: int = 20
    ) -> 'FeatureEngineer':
        """
        Encode categorical features
        
        Args:
            columns: List of categorical columns to encode
            method: Encoding method ('onehot', 'label', 'target')
            max_categories: Maximum categories for one-hot encoding
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Encoding categorical features using {method} method...")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            n_categories = self.df[col].nunique()
            
            if method == 'onehot' and n_categories <= max_categories:
                # One-hot encoding for low cardinality features
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.logger.info(f"One-hot encoded {col} ({n_categories} categories)")
                
            elif method == 'label' or n_categories > max_categories:
                # Label encoding for high cardinality features
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
                self.logger.info(f"Label encoded {col} ({n_categories} categories)")
        
        return self
    
    def scale_numerical_features(
        self,
        columns: List[str],
        method: str = 'standard'
    ) -> 'FeatureEngineer':
        """
        Scale numerical features
        
        Args:
            columns: List of numerical columns to scale
            method: Scaling method ('standard', 'minmax')
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Scaling numerical features using {method} method...")
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                self.logger.warning(f"Unknown scaling method: {method}")
                continue
            
            self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler
            self.logger.info(f"Scaled {col}")
        
        return self
    
    def select_features(
        self,
        target_column: str,
        k: int = 20,
        method: str = 'f_regression'
    ) -> Tuple['FeatureEngineer', List[str]]:
        """
        Select top k features based on statistical tests
        
        Args:
            target_column: Name of target variable
            k: Number of top features to select
            method: Selection method ('f_regression', 'mutual_info')
            
        Returns:
            Tuple of (Self, list of selected feature names)
        """
        self.logger.info(f"Selecting top {k} features using {method}...")
        
        # Get numerical features only
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from features
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        # Prepare data
        X = self.df[numerical_cols].fillna(0)
        y = self.df[target_column].fillna(0)
        
        # Select features
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, len(numerical_cols)))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(numerical_cols)))
        else:
            self.logger.warning(f"Unknown method: {method}")
            return self, numerical_cols
        
        selector.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.logger.info(f"Selected features: {selected_features}")
        
        return self, selected_features
    
    def get_engineered_data(self) -> pd.DataFrame:
        """
        Get DataFrame with engineered features
        
        Returns:
            DataFrame with new features
        """
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names
        
        Returns:
            List of feature names
        """
        return self.df.columns.tolist()


def engineer_sparcs_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to engineer features for SPARCS data
    
    Args:
        df: Cleaned SPARCS DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Initialize engineer
    engineer = FeatureEngineer(df)
    
    # Create features
    engineered_df = (engineer
        .create_age_features()
        .create_length_of_stay_features()
        .create_cost_features()
        .create_severity_features()
        .create_admission_features()
        .get_engineered_data()
    )
    
    logger.info("Feature engineering completed!")
    logger.info(f"Original features: {len(df.columns)}")
    logger.info(f"Engineered features: {len(engineered_df.columns)}")
    logger.info(f"New features added: {len(engineered_df.columns) - len(df.columns)}")
    
    return engineered_df


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from clean_data import clean_sparcs_data
    
    # Load and clean data
    loader = DataLoader()
    df = loader.load_from_ny_health_api(limit=5000)
    cleaned_df = clean_sparcs_data(df)
    
    # Engineer features
    engineered_df = engineer_sparcs_features(cleaned_df)
    
    print(f"\nEngineered data shape: {engineered_df.shape}")
    print(f"\nNew columns:")
    print([col for col in engineered_df.columns if col not in df.columns])
