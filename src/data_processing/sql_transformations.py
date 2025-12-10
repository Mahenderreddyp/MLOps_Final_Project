"""
SQL Transformations Module for Hospital Cost Prediction

This module handles SQL-based feature engineering operations for Azure ML.
It provides functions to transform data using SQL queries for efficient
feature creation and data manipulation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from azureml.data import TabularDataset
from azureml.data.datapath import DataPath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLTransformer:
    """Class to handle SQL-based transformations"""
    
    def __init__(self, dataset: Optional[TabularDataset] = None):
        """
        Initialize SQLTransformer
        
        Args:
            dataset: Azure ML TabularDataset (optional)
        """
        self.dataset = dataset
        self.logger = logger
        self.transformation_history = []
    
    def create_age_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group features using SQL-like transformations
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with age group features
        """
        self.logger.info("Creating age group features...")
        
        if 'age_group' in df.columns:
            # Create age risk score
            age_risk_mapping = {
                '0-17': 1,
                '18-29': 2,
                '30-49': 3,
                '50-69': 4,
                '70 or Older': 5,
                '70+': 5
            }
            
            df['age_risk_score'] = df['age_group'].map(age_risk_mapping)
            
            # Create binary flags
            df['is_pediatric'] = (df['age_group'] == '0-17').astype(int)
            df['is_elderly'] = (df['age_group'].isin(['70 or Older', '70+'])).astype(int)
            df['is_middle_aged'] = (df['age_group'] == '50-69').astype(int)
            
            self.transformation_history.append('age_group_features')
        
        return df
    
    def create_cost_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cost ratio features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cost ratio features
        """
        self.logger.info("Creating cost ratio features...")
        
        # Ensure numeric
        numeric_cols = ['total_costs', 'total_charges', 'length_of_stay']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cost per day
        if 'total_costs' in df.columns and 'length_of_stay' in df.columns:
            df['cost_per_day'] = df['total_costs'] / (df['length_of_stay'] + 1)
        
        # Cost to charge ratio
        if 'total_costs' in df.columns and 'total_charges' in df.columns:
            df['cost_to_charge_ratio'] = df['total_costs'] / (df['total_charges'] + 1)
        
        # Cost efficiency (inverse of LOS)
        if 'total_costs' in df.columns and 'length_of_stay' in df.columns:
            df['cost_efficiency'] = 1 / (df['length_of_stay'] + 1) * df['total_costs']
        
        self.transformation_history.append('cost_ratio_features')
        
        return df
    
    def create_severity_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between severity and other variables
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Creating severity interaction features...")
        
        # Severity encoding
        if 'apr_severity_of_illness_description' in df.columns:
            severity_map = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}
            df['severity_encoded'] = df['apr_severity_of_illness_description'].map(severity_map)
        
        # Risk of mortality encoding
        if 'apr_risk_of_mortality' in df.columns:
            mortality_map = {'Minor': 1, 'Moderate': 2, 'Major': 3, 'Extreme': 4}
            df['mortality_risk_encoded'] = df['apr_risk_of_mortality'].map(mortality_map)
        
        # Interaction: Severity * LOS
        if 'severity_encoded' in df.columns and 'length_of_stay' in df.columns:
            df['severity_los_interaction'] = df['severity_encoded'] * df['length_of_stay']
        
        # Interaction: Severity * Age
        if 'severity_encoded' in df.columns and 'age_risk_score' in df.columns:
            df['severity_age_interaction'] = df['severity_encoded'] * df['age_risk_score']
        
        # Combined risk score
        if 'severity_encoded' in df.columns and 'mortality_risk_encoded' in df.columns:
            df['combined_risk_score'] = (
                df['severity_encoded'] * 0.6 + 
                df['mortality_risk_encoded'] * 0.4
            )
        
        self.transformation_history.append('severity_interaction_features')
        
        return df
    
    def create_admission_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create admission-related features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with admission features
        """
        self.logger.info("Creating admission features...")
        
        # Emergency admission flag
        if 'type_of_admission' in df.columns:
            df['is_emergency'] = (
                df['type_of_admission'].str.contains('Emergency', case=False, na=False)
            ).astype(int)
            
            df['is_elective'] = (
                df['type_of_admission'].str.contains('Elective', case=False, na=False)
            ).astype(int)
        
        # ED indicator
        if 'emergency_department_indicator' in df.columns:
            df['ed_visit'] = (df['emergency_department_indicator'] == 'Y').astype(int)
        
        # Urgent admission
        if 'type_of_admission' in df.columns:
            df['is_urgent'] = (
                df['type_of_admission'].str.contains('Urgent', case=False, na=False)
            ).astype(int)
        
        self.transformation_history.append('admission_features')
        
        return df
    
    def create_medical_surgical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create medical/surgical classification features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with medical/surgical features
        """
        self.logger.info("Creating medical/surgical features...")
        
        if 'apr_medical_surgical_description' in df.columns:
            df['is_surgical'] = (
                df['apr_medical_surgical_description'].str.contains('Surgical', case=False, na=False)
            ).astype(int)
            
            df['is_medical'] = (
                df['apr_medical_surgical_description'].str.contains('Medical', case=False, na=False)
            ).astype(int)
        
        self.transformation_history.append('medical_surgical_features')
        
        return df
    
    def create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create payment typology features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with payment features
        """
        self.logger.info("Creating payment features...")
        
        if 'payment_typology_1' in df.columns:
            # Medicare flag
            df['is_medicare'] = (
                df['payment_typology_1'].str.contains('Medicare', case=False, na=False)
            ).astype(int)
            
            # Medicaid flag
            df['is_medicaid'] = (
                df['payment_typology_1'].str.contains('Medicaid', case=False, na=False)
            ).astype(int)
            
            # Private insurance flag
            df['is_private'] = (
                df['payment_typology_1'].str.contains('Private', case=False, na=False)
            ).astype(int)
        
        self.transformation_history.append('payment_features')
        
        return df
    
    def create_los_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create length of stay categories
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with LOS categories
        """
        self.logger.info("Creating LOS categories...")
        
        if 'length_of_stay' in df.columns:
            df['length_of_stay'] = pd.to_numeric(df['length_of_stay'], errors='coerce')
            
            # Create categories
            df['los_category'] = pd.cut(
                df['length_of_stay'],
                bins=[-1, 1, 3, 7, 14, np.inf],
                labels=['Very Short (0-1)', 'Short (2-3)', 'Medium (4-7)', 
                       'Long (8-14)', 'Very Long (15+)']
            )
            
            # Binary flags
            df['extended_stay'] = (df['length_of_stay'] > 7).astype(int)
            df['short_stay'] = (df['length_of_stay'] <= 3).astype(int)
        
        self.transformation_history.append('los_categories')
        
        return df
    
    def apply_sql_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all SQL transformations in sequence
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        self.logger.info("Applying SQL transformations...")
        
        df = self.create_age_group_features(df)
        df = self.create_cost_ratio_features(df)
        df = self.create_severity_interaction_features(df)
        df = self.create_admission_features(df)
        df = self.create_medical_surgical_features(df)
        df = self.create_payment_features(df)
        df = self.create_los_categories(df)
        
        self.logger.info(f"Applied {len(self.transformation_history)} transformation groups")
        self.logger.info(f"Transformations: {', '.join(self.transformation_history)}")
        
        return df
    
    def get_transformation_sql(self, transformation_name: str) -> str:
        """
        Get SQL query for a specific transformation (for reference)
        
        Args:
            transformation_name: Name of transformation
            
        Returns:
            SQL query string
        """
        sql_queries = {
            'age_group_features': """
                ALTER TABLE data ADD COLUMN age_risk_score INT;
                UPDATE data SET age_risk_score = 
                    CASE age_group
                        WHEN '0-17' THEN 1
                        WHEN '18-29' THEN 2
                        WHEN '30-49' THEN 3
                        WHEN '50-69' THEN 4
                        WHEN '70+' THEN 5
                        ELSE NULL
                    END;
            """,
            'cost_ratio_features': """
                ALTER TABLE data ADD COLUMN cost_per_day FLOAT;
                UPDATE data SET cost_per_day = 
                    total_costs / NULLIF(length_of_stay + 1, 0);
                
                ALTER TABLE data ADD COLUMN cost_to_charge_ratio FLOAT;
                UPDATE data SET cost_to_charge_ratio = 
                    total_costs / NULLIF(total_charges + 1, 0);
            """,
            'severity_interaction_features': """
                ALTER TABLE data ADD COLUMN severity_encoded INT;
                UPDATE data SET severity_encoded = 
                    CASE apr_severity_of_illness_description
                        WHEN 'Minor' THEN 1
                        WHEN 'Moderate' THEN 2
                        WHEN 'Major' THEN 3
                        WHEN 'Extreme' THEN 4
                        ELSE NULL
                    END;
                
                ALTER TABLE data ADD COLUMN severity_los_interaction FLOAT;
                UPDATE data SET severity_los_interaction = 
                    severity_encoded * length_of_stay;
            """
        }
        
        return sql_queries.get(transformation_name, "-- Transformation not found")


def apply_sql_transformations_to_sparcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to apply SQL transformations to SPARCS data
    
    Args:
        df: SPARCS DataFrame
        
    Returns:
        Transformed DataFrame
    """
    logger.info("Starting SQL transformations on SPARCS data...")
    
    transformer = SQLTransformer()
    transformed_df = transformer.apply_sql_transformations(df)
    
    logger.info(f"Original columns: {len(df.columns)}")
    logger.info(f"Transformed columns: {len(transformed_df.columns)}")
    logger.info(f"New columns added: {len(transformed_df.columns) - len(df.columns)}")
    
    return transformed_df


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    # Load sample data
    loader = DataLoader()
    df = loader.load_from_ny_health_api(limit=1000)
    
    # Apply SQL transformations
    transformed_df = apply_sql_transformations_to_sparcs(df)
    
    print(f"\nTransformed data shape: {transformed_df.shape}")
    print(f"\nNew columns:")
    new_cols = [col for col in transformed_df.columns if col not in df.columns]
    print(new_cols)

