"""
Data Drift Detection Script

This script detects data drift between baseline and current datasets.
Uses statistical tests (KS test, PSI) to identify distribution changes.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI)
    
    Args:
        expected: Baseline distribution
        actual: Current distribution
        buckets: Number of bins for discretization
        
    Returns:
        PSI value
    """
    # Remove NaN values
    expected = expected.dropna()
    actual = actual.dropna()
    
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    
    # Create bins based on expected distribution
    breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Calculate expected and actual distributions
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi


def calculate_ks_test(baseline: pd.Series, current: pd.Series) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov test statistic and p-value
    
    Args:
        baseline: Baseline distribution
        current: Current distribution
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    # Remove NaN values
    baseline = baseline.dropna()
    current = current.dropna()
    
    if len(baseline) == 0 or len(current) == 0:
        return np.nan, np.nan
    
    # Perform KS test
    statistic, pvalue = stats.ks_2samp(baseline, current)
    
    return statistic, pvalue


def detect_drift_for_feature(
    baseline: pd.Series,
    current: pd.Series,
    feature_name: str,
    drift_threshold: float = 0.1
) -> Dict:
    """
    Detect drift for a single feature
    
    Args:
        baseline: Baseline feature values
        current: Current feature values
        feature_name: Name of the feature
        drift_threshold: PSI threshold for drift detection
        
    Returns:
        Dictionary with drift detection results
    """
    results = {
        'feature': feature_name,
        'drift_detected': False,
        'psi': np.nan,
        'ks_statistic': np.nan,
        'ks_pvalue': np.nan,
        'baseline_mean': np.nan,
        'current_mean': np.nan,
        'baseline_std': np.nan,
        'current_std': np.nan
    }
    
    try:
        # Calculate statistics
        results['baseline_mean'] = float(baseline.mean())
        results['current_mean'] = float(current.mean())
        results['baseline_std'] = float(baseline.std())
        results['current_std'] = float(current.std())
        
        # Calculate PSI
        psi = calculate_psi(baseline, current)
        results['psi'] = float(psi)
        
        # Calculate KS test
        ks_stat, ks_pvalue = calculate_ks_test(baseline, current)
        results['ks_statistic'] = float(ks_stat)
        results['ks_pvalue'] = float(ks_pvalue)
        
        # Determine drift
        # PSI thresholds: < 0.1 (no drift), 0.1-0.25 (minor), > 0.25 (significant)
        if psi > drift_threshold or ks_pvalue < 0.05:
            results['drift_detected'] = True
        
    except Exception as e:
        logger.warning(f"Error detecting drift for {feature_name}: {str(e)}")
    
    return results


def detect_data_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_threshold: float = 0.1,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Detect data drift between baseline and current datasets
    
    Args:
        baseline_df: Baseline DataFrame
        current_df: Current DataFrame
        drift_threshold: PSI threshold for drift detection
        features: List of features to check (all numeric if None)
        
    Returns:
        DataFrame with drift detection results
    """
    logger.info("Starting data drift detection...")
    
    # Select numeric features
    if features is None:
        numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col in current_df.columns]
    
    logger.info(f"Checking drift for {len(features)} features...")
    
    # Detect drift for each feature
    drift_results = []
    
    for feature in features:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            logger.warning(f"Feature {feature} not found in both datasets, skipping...")
            continue
        
        baseline_series = baseline_df[feature]
        current_series = current_df[feature]
        
        result = detect_drift_for_feature(
            baseline_series,
            current_series,
            feature,
            drift_threshold
        )
        
        drift_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(drift_results)
    
    # Sort by PSI (highest drift first)
    results_df = results_df.sort_values('psi', ascending=False, na_last=True)
    
    # Summary
    num_drifted = results_df['drift_detected'].sum()
    logger.info(f"Drift detected in {num_drifted} out of {len(results_df)} features")
    
    return results_df


def print_drift_summary(results_df: pd.DataFrame) -> None:
    """Print drift detection summary"""
    print("\n" + "="*80)
    print("DATA DRIFT DETECTION SUMMARY")
    print("="*80)
    
    total_features = len(results_df)
    drifted_features = results_df['drift_detected'].sum()
    
    print(f"\nTotal features analyzed: {total_features}")
    print(f"Features with drift detected: {drifted_features}")
    print(f"Features without drift: {total_features - drifted_features}")
    
    if drifted_features > 0:
        print("\n" + "-"*80)
        print("TOP FEATURES WITH DRIFT (sorted by PSI):")
        print("-"*80)
        
        drifted = results_df[results_df['drift_detected']].head(10)
        
        for idx, row in drifted.iterrows():
            print(f"\n{row['feature']}:")
            print(f"  PSI: {row['psi']:.4f}")
            print(f"  KS Statistic: {row['ks_statistic']:.4f}")
            print(f"  KS P-value: {row['ks_pvalue']:.6f}")
            print(f"  Baseline Mean: {row['baseline_mean']:.2f}")
            print(f"  Current Mean: {row['current_mean']:.2f}")
            print(f"  Mean Change: {((row['current_mean'] - row['baseline_mean']) / row['baseline_mean'] * 100):.2f}%")
    
    print("="*80 + "\n")


def main():
    """Main function to detect data drift"""
    
    parser = argparse.ArgumentParser(description='Detect Data Drift')
    
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Path to baseline dataset (CSV)'
    )
    
    parser.add_argument(
        '--current',
        type=str,
        required=True,
        help='Path to current dataset (CSV)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='drift_results.csv',
        help='Output path for drift results'
    )
    
    parser.add_argument(
        '--output-json',
        type=str,
        default='drift_results.json',
        help='Output path for drift results (JSON)'
    )
    
    parser.add_argument(
        '--drift-threshold',
        type=float,
        default=0.1,
        help='PSI threshold for drift detection (default: 0.1)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        help='Specific features to check (all numeric if not specified)'
    )
    
    args = parser.parse_args()
    
    # Load datasets
    logger.info(f"Loading baseline dataset from {args.baseline}...")
    baseline_df = pd.read_csv(args.baseline)
    logger.info(f"Baseline dataset: {len(baseline_df)} rows, {len(baseline_df.columns)} columns")
    
    logger.info(f"Loading current dataset from {args.current}...")
    current_df = pd.read_csv(args.current)
    logger.info(f"Current dataset: {len(current_df)} rows, {len(current_df.columns)} columns")
    
    # Detect drift
    results_df = detect_data_drift(
        baseline_df=baseline_df,
        current_df=current_df,
        drift_threshold=args.drift_threshold,
        features=args.features
    )
    
    # Save results
    logger.info(f"Saving results to {args.output}...")
    results_df.to_csv(args.output, index=False)
    
    # Save JSON summary
    summary = {
        'total_features': len(results_df),
        'drifted_features': int(results_df['drift_detected'].sum()),
        'drift_threshold': args.drift_threshold,
        'top_drifted_features': results_df[results_df['drift_detected']].head(10).to_dict('records')
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"JSON summary saved to {args.output_json}")
    
    # Print summary
    print_drift_summary(results_df)
    
    logger.info("Data drift detection completed!")


if __name__ == "__main__":
    main()

