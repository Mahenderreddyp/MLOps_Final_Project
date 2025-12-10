"""
Model Monitoring Setup Script

This script sets up monitoring for the hospital cost prediction model.
It configures data drift detection, performance tracking, and alerting.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional
from azureml.core import Workspace, Model
from azureml.datadrift import DataDriftDetector
from azureml.monitoring import ModelDataCollector
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_workspace(
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None
) -> Workspace:
    """
    Get Azure ML Workspace
    
    Args:
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        workspace_name: Workspace name
        
    Returns:
        Azure ML Workspace object
    """
    # Try to load from config file first
    try:
        ws = Workspace.from_config()
        logger.info(f"Loaded workspace from config: {ws.name}")
        return ws
    except Exception:
        pass
    
    # Use provided parameters or environment variables
    subscription_id = subscription_id or os.getenv('AZURE_SUBSCRIPTION_ID')
    resource_group = resource_group or os.getenv('AZURE_RESOURCE_GROUP')
    workspace_name = workspace_name or os.getenv('AZURE_WORKSPACE_NAME')
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Workspace credentials not found. Provide subscription_id, "
            "resource_group, and workspace_name, or set environment variables."
        )
    
    logger.info(f"Connecting to workspace: {workspace_name}")
    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name
    )
    
    logger.info(f"Connected to workspace: {ws.name}")
    return ws


def setup_data_drift_detector(
    workspace: Workspace,
    model_name: str,
    baseline_dataset_name: str,
    target_dataset_name: str,
    detector_name: str = "hospital-cost-drift-detector",
    schedule_frequency: str = "Weekly"
) -> DataDriftDetector:
    """
    Set up data drift detector
    
    Args:
        workspace: Azure ML Workspace
        model_name: Model name to monitor
        baseline_dataset_name: Baseline dataset name
        target_dataset_name: Target dataset name for comparison
        detector_name: Detector name
        schedule_frequency: Schedule frequency ('Daily', 'Weekly', 'Monthly')
        
    Returns:
        DataDriftDetector object
    """
    logger.info(f"Setting up data drift detector: {detector_name}")
    
    # Get model
    model = Model(workspace, name=model_name)
    logger.info(f"Found model: {model_name}")
    
    # Get datasets
    from azureml.core import Dataset
    baseline_dataset = Dataset.get_by_name(workspace, baseline_dataset_name)
    target_dataset = Dataset.get_by_name(workspace, target_dataset_name)
    
    logger.info(f"Baseline dataset: {baseline_dataset_name}")
    logger.info(f"Target dataset: {target_dataset_name}")
    
    # Create data drift detector
    detector = DataDriftDetector.create_from_datasets(
        workspace=workspace,
        name=detector_name,
        baseline_data_dataset=baseline_dataset,
        target_data_dataset=target_dataset,
        compute_target=None,  # Will use default compute
        frequency=schedule_frequency,
        feature_list=None,  # Monitor all features
        drift_threshold=0.1,  # 10% drift threshold
        latency=24  # 24 hours latency
    )
    
    logger.info(f"Data drift detector created: {detector_name}")
    
    return detector


def setup_model_data_collector(
    workspace: Workspace,
    model_name: str,
    collector_name: str = "hospital-cost-collector"
) -> ModelDataCollector:
    """
    Set up model data collector for monitoring
    
    Args:
        workspace: Azure ML Workspace
        model_name: Model name
        collector_name: Collector name
        
    Returns:
        ModelDataCollector object
    """
    logger.info(f"Setting up model data collector: {collector_name}")
    
    # Get model
    model = Model(workspace, name=model_name)
    
    # Create data collector
    collector = ModelDataCollector(
        model_name=model_name,
        model_version=model.version,
        collector_name=collector_name,
        workspace=workspace
    )
    
    logger.info(f"Model data collector created: {collector_name}")
    
    return collector


def create_monitoring_config(
    output_path: str = "monitoring_config.json",
    model_name: str = "hospital-cost-predictor",
    baseline_dataset: str = "sparcs_baseline",
    target_dataset: str = "sparcs_current",
    drift_threshold: float = 0.1,
    performance_threshold_r2: float = 0.75,
    alert_email: str = None
) -> None:
    """
    Create monitoring configuration file
    
    Args:
        output_path: Path to save config
        model_name: Model name
        baseline_dataset: Baseline dataset name
        target_dataset: Target dataset name
        drift_threshold: Data drift threshold
        performance_threshold_r2: Minimum R² score threshold
        alert_email: Email for alerts
    """
    config = {
        "monitoring": {
            "model_name": model_name,
            "baseline_dataset": baseline_dataset,
            "target_dataset": target_dataset,
            "drift_detection": {
                "enabled": True,
                "threshold": drift_threshold,
                "schedule": "Weekly",
                "methods": ["ks_test", "psi"]
            },
            "performance_monitoring": {
                "enabled": True,
                "metrics": ["r2_score", "mae", "rmse"],
                "thresholds": {
                    "r2_score": performance_threshold_r2,
                    "mae": None,
                    "rmse": None
                },
                "check_frequency": "Daily"
            },
            "alerts": {
                "enabled": True,
                "email": alert_email,
                "drift_alert": True,
                "performance_alert": True
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Monitoring configuration saved to: {output_path}")


def main():
    """Main function to set up monitoring"""
    
    parser = argparse.ArgumentParser(description='Set up Model Monitoring')
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='hospital-cost-predictor',
        help='Model name to monitor'
    )
    
    parser.add_argument(
        '--baseline-dataset',
        type=str,
        default='sparcs_baseline',
        help='Baseline dataset name'
    )
    
    parser.add_argument(
        '--target-dataset',
        type=str,
        default='sparcs_current',
        help='Target dataset name'
    )
    
    parser.add_argument(
        '--detector-name',
        type=str,
        default='hospital-cost-drift-detector',
        help='Drift detector name'
    )
    
    parser.add_argument(
        '--schedule',
        type=str,
        default='Weekly',
        choices=['Daily', 'Weekly', 'Monthly'],
        help='Monitoring schedule frequency'
    )
    
    parser.add_argument(
        '--config-output',
        type=str,
        default='monitoring_config.json',
        help='Output path for monitoring config'
    )
    
    parser.add_argument(
        '--subscription-id',
        type=str,
        help='Azure subscription ID'
    )
    
    parser.add_argument(
        '--resource-group',
        type=str,
        help='Azure resource group'
    )
    
    parser.add_argument(
        '--workspace-name',
        type=str,
        help='Azure ML workspace name'
    )
    
    parser.add_argument(
        '--create-config-only',
        action='store_true',
        help='Only create config file, do not set up Azure resources'
    )
    
    args = parser.parse_args()
    
    # Create monitoring configuration
    create_monitoring_config(
        output_path=args.config_output,
        model_name=args.model_name,
        baseline_dataset=args.baseline_dataset,
        target_dataset=args.target_dataset
    )
    
    if args.create_config_only:
        logger.info("Configuration file created. Skipping Azure setup.")
        return
    
    # Get workspace
    workspace = get_workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    # Set up data drift detector
    try:
        detector = setup_data_drift_detector(
            workspace=workspace,
            model_name=args.model_name,
            baseline_dataset_name=args.baseline_dataset,
            target_dataset_name=args.target_dataset,
            detector_name=args.detector_name,
            schedule_frequency=args.schedule
        )
        logger.info(f"Data drift detector set up successfully: {detector.name}")
    except Exception as e:
        logger.warning(f"Could not set up data drift detector: {str(e)}")
    
    # Set up model data collector
    try:
        collector = setup_model_data_collector(
            workspace=workspace,
            model_name=args.model_name
        )
        logger.info("Model data collector set up successfully")
    except Exception as e:
        logger.warning(f"Could not set up model data collector: {str(e)}")
    
    logger.info("Monitoring setup completed!")
    logger.info(f"Configuration saved to: {args.config_output}")


if __name__ == "__main__":
    main()

