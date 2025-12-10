"""
Run Azure ML Pipeline Script

This script runs the Azure ML pipeline for hospital cost prediction.
It can run existing published pipelines or create and run new ones.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from azureml.core import Workspace, Experiment
from azureml.pipeline.core import PublishedPipeline
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.create_pipeline import HospitalCostPipeline

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
    Get or create Azure ML Workspace
    
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


def run_published_pipeline(
    workspace: Workspace,
    pipeline_id: str = None,
    pipeline_name: str = None,
    experiment_name: str = "hospital-cost-prediction"
) -> None:
    """
    Run a published pipeline
    
    Args:
        workspace: Azure ML Workspace
        pipeline_id: Published pipeline ID
        pipeline_name: Published pipeline name (if ID not provided)
        experiment_name: Experiment name
    """
    logger.info("Running published pipeline...")
    
    # Get published pipeline
    if pipeline_id:
        published_pipeline = PublishedPipeline.get(workspace, id=pipeline_id)
    elif pipeline_name:
        # Get latest version by name
        published_pipelines = PublishedPipeline.list(workspace)
        matching = [p for p in published_pipelines if p.name == pipeline_name]
        if not matching:
            raise ValueError(f"Published pipeline '{pipeline_name}' not found")
        published_pipeline = matching[0]
    else:
        raise ValueError("Either pipeline_id or pipeline_name must be provided")
    
    logger.info(f"Found published pipeline: {published_pipeline.name} (ID: {published_pipeline.id})")
    
    # Create experiment
    experiment = Experiment(workspace, experiment_name)
    
    # Submit pipeline run
    logger.info(f"Submitting pipeline to experiment: {experiment_name}")
    pipeline_run = experiment.submit(published_pipeline)
    
    logger.info(f"Pipeline submitted. Run ID: {pipeline_run.id}")
    logger.info(f"Run URL: {pipeline_run.get_portal_url()}")
    
    # Wait for completion
    logger.info("Waiting for pipeline to complete...")
    pipeline_run.wait_for_completion(show_output=True)
    
    logger.info("Pipeline completed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE RUN SUMMARY")
    print("="*60)
    print(f"Run ID: {pipeline_run.id}")
    print(f"Status: {pipeline_run.get_status()}")
    print(f"Portal URL: {pipeline_run.get_portal_url()}")
    print("="*60 + "\n")


def run_new_pipeline(
    workspace: Workspace,
    dataset_name: str = "sparcs_raw",
    experiment_name: str = "hospital-cost-prediction",
    compute_name: str = "cpu-cluster"
) -> None:
    """
    Create and run a new pipeline
    
    Args:
        workspace: Azure ML Workspace
        dataset_name: Name of registered dataset
        experiment_name: Experiment name
        compute_name: Compute target name
    """
    logger.info("Creating and running new pipeline...")
    
    # Get dataset
    try:
        dataset = workspace.datasets[dataset_name]
        logger.info(f"Found dataset: {dataset_name}")
    except KeyError:
        logger.warning(f"Dataset '{dataset_name}' not found. Please register it first.")
        raise
    
    # Create pipeline
    pipeline_builder = HospitalCostPipeline(
        workspace=workspace,
        compute_name=compute_name,
        experiment_name=experiment_name
    )
    
    pipeline = pipeline_builder.build_pipeline(dataset)
    
    # Run pipeline
    pipeline_builder.run_pipeline(pipeline)
    
    logger.info("Pipeline run completed!")


def main():
    """Main function to run pipeline"""
    
    parser = argparse.ArgumentParser(description='Run Azure ML Pipeline for Hospital Cost Prediction')
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='hospital-cost-prediction',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--pipeline-id',
        type=str,
        help='Published pipeline ID to run'
    )
    
    parser.add_argument(
        '--pipeline-name',
        type=str,
        help='Published pipeline name to run'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='sparcs_raw',
        help='Dataset name (for new pipeline)'
    )
    
    parser.add_argument(
        '--compute-name',
        type=str,
        default='cpu-cluster',
        help='Compute target name'
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
    
    args = parser.parse_args()
    
    # Get workspace
    workspace = get_workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    # Run pipeline
    if args.pipeline_id or args.pipeline_name:
        # Run published pipeline
        run_published_pipeline(
            workspace=workspace,
            pipeline_id=args.pipeline_id,
            pipeline_name=args.pipeline_name,
            experiment_name=args.experiment_name
        )
    else:
        # Create and run new pipeline
        run_new_pipeline(
            workspace=workspace,
            dataset_name=args.dataset_name,
            experiment_name=args.experiment_name,
            compute_name=args.compute_name
        )


if __name__ == "__main__":
    main()

