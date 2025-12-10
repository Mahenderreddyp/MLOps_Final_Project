"""
Model Deployment Script for Hospital Cost Prediction

This script deploys trained models to Azure ML endpoints for production use.
Supports deployment to ACI (Azure Container Instances) and AKS (Azure Kubernetes Service).
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, AksWebservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.model import Model
import json

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


def get_model(
    workspace: Workspace,
    model_name: str,
    version: int = None
) -> Model:
    """
    Get registered model from workspace
    
    Args:
        workspace: Azure ML Workspace
        model_name: Model name
        version: Model version (latest if None)
        
    Returns:
        Model object
    """
    if version:
        model = Model(workspace, name=model_name, version=version)
        logger.info(f"Found model: {model_name} (version {version})")
    else:
        model = Model(workspace, name=model_name)
        logger.info(f"Found model: {model_name} (latest version)")
    
    return model


def create_inference_config(
    scoring_script: str = "score.py",
    environment: Environment = None
) -> InferenceConfig:
    """
    Create inference configuration
    
    Args:
        scoring_script: Path to scoring script
        environment: Environment object (creates default if None)
        
    Returns:
        InferenceConfig object
    """
    logger.info(f"Creating inference config with scoring script: {scoring_script}")
    
    # Create default environment if not provided
    if environment is None:
        env = Environment(name="hospital-cost-env")
        env.python.conda_dependencies.add_pip_package("pandas==2.0.3")
        env.python.conda_dependencies.add_pip_package("numpy==1.24.3")
        env.python.conda_dependencies.add_pip_package("scikit-learn==1.3.0")
        env.python.conda_dependencies.add_pip_package("joblib==1.3.1")
        env.python.conda_dependencies.add_pip_package("azureml-defaults")
    else:
        env = environment
    
    inference_config = InferenceConfig(
        entry_script=scoring_script,
        environment=env
    )
    
    return inference_config


def deploy_to_aci(
    workspace: Workspace,
    model: Model,
    inference_config: InferenceConfig,
    deployment_name: str,
    cpu_cores: int = 1,
    memory_gb: int = 2
) -> AciWebservice:
    """
    Deploy model to Azure Container Instances (ACI)
    
    Args:
        workspace: Azure ML Workspace
        model: Model to deploy
        inference_config: Inference configuration
        deployment_name: Deployment name
        cpu_cores: Number of CPU cores
        memory_gb: Memory in GB
        
    Returns:
        AciWebservice object
    """
    logger.info(f"Deploying model to ACI: {deployment_name}")
    
    # Create deployment configuration
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb
    )
    
    # Deploy service
    service = Model.deploy(
        workspace=workspace,
        name=deployment_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config
    )
    
    logger.info("Waiting for deployment to complete...")
    service.wait_for_deployment(show_output=True)
    
    logger.info(f"Deployment completed! Service URL: {service.scoring_uri}")
    
    return service


def deploy_to_aks(
    workspace: Workspace,
    model: Model,
    inference_config: InferenceConfig,
    deployment_name: str,
    compute_target_name: str,
    cpu_cores: int = 1,
    memory_gb: int = 2,
    instance_count: int = 3
) -> AksWebservice:
    """
    Deploy model to Azure Kubernetes Service (AKS)
    
    Args:
        workspace: Azure ML Workspace
        model: Model to deploy
        inference_config: Inference configuration
        deployment_name: Deployment name
        compute_target_name: AKS compute target name
        cpu_cores: Number of CPU cores per instance
        memory_gb: Memory in GB per instance
        instance_count: Number of instances
        
    Returns:
        AksWebservice object
    """
    logger.info(f"Deploying model to AKS: {deployment_name}")
    
    # Get AKS compute target
    from azureml.core.compute import AksCompute
    aks_target = AksCompute(workspace, name=compute_target_name)
    
    # Create deployment configuration
    deployment_config = AksWebservice.deploy_configuration(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        enable_app_insights=True
    )
    
    # Deploy service
    service = Model.deploy(
        workspace=workspace,
        name=deployment_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        deployment_target=aks_target
    )
    
    logger.info("Waiting for deployment to complete...")
    service.wait_for_deployment(show_output=True)
    
    logger.info(f"Deployment completed! Service URL: {service.scoring_uri}")
    
    return service


def create_scoring_script(output_path: str = "score.py") -> None:
    """
    Create scoring script template
    
    Args:
        output_path: Path to save scoring script
    """
    scoring_script_content = '''"""
Scoring script for Hospital Cost Prediction model
"""

import json
import numpy as np
import pandas as pd
import joblib
import os


def init():
    """Initialize model"""
    global model
    
    # Get model path
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    
    # Load model
    model = joblib.load(model_path)
    print("Model loaded successfully")


def run(raw_data):
    """Make predictions"""
    try:
        # Parse input data
        data = json.loads(raw_data)['data']
        df = pd.DataFrame(data)
        
        # Select numeric features only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Inverse log transform if needed
        predictions = np.expm1(predictions)
        
        # Return results
        return json.dumps({"predictions": predictions.tolist()})
    
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
'''
    
    with open(output_path, 'w') as f:
        f.write(scoring_script_content)
    
    logger.info(f"Scoring script created at: {output_path}")


def main():
    """Main function to deploy model"""
    
    parser = argparse.ArgumentParser(description='Deploy Hospital Cost Prediction Model')
    
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help='Registered model name'
    )
    
    parser.add_argument(
        '--version',
        type=int,
        help='Model version (latest if not specified)'
    )
    
    parser.add_argument(
        '--deployment-name',
        type=str,
        default='hospital-cost-service',
        help='Deployment name'
    )
    
    parser.add_argument(
        '--compute-target',
        type=str,
        default='aci',
        choices=['aci', 'aks'],
        help='Deployment target (aci or aks)'
    )
    
    parser.add_argument(
        '--aks-cluster',
        type=str,
        help='AKS cluster name (required for AKS deployment)'
    )
    
    parser.add_argument(
        '--scoring-script',
        type=str,
        default='score.py',
        help='Path to scoring script'
    )
    
    parser.add_argument(
        '--cpu-cores',
        type=int,
        default=1,
        help='Number of CPU cores'
    )
    
    parser.add_argument(
        '--memory-gb',
        type=int,
        default=2,
        help='Memory in GB'
    )
    
    parser.add_argument(
        '--instance-count',
        type=int,
        default=3,
        help='Number of instances (for AKS)'
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
        '--create-scoring-script',
        action='store_true',
        help='Create scoring script template'
    )
    
    args = parser.parse_args()
    
    # Create scoring script if requested
    if args.create_scoring_script or not os.path.exists(args.scoring_script):
        logger.info("Creating scoring script...")
        create_scoring_script(args.scoring_script)
    
    # Get workspace
    workspace = get_workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    # Get model
    model = get_model(workspace, args.model_name, args.version)
    
    # Create inference config
    inference_config = create_inference_config(args.scoring_script)
    
    # Deploy model
    if args.compute_target == 'aci':
        service = deploy_to_aci(
            workspace=workspace,
            model=model,
            inference_config=inference_config,
            deployment_name=args.deployment_name,
            cpu_cores=args.cpu_cores,
            memory_gb=args.memory_gb
        )
    else:  # aks
        if not args.aks_cluster:
            raise ValueError("--aks-cluster is required for AKS deployment")
        
        service = deploy_to_aks(
            workspace=workspace,
            model=model,
            inference_config=inference_config,
            deployment_name=args.deployment_name,
            compute_target_name=args.aks_cluster,
            cpu_cores=args.cpu_cores,
            memory_gb=args.memory_gb,
            instance_count=args.instance_count
        )
    
    # Print deployment info
    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)
    print(f"Service Name: {service.name}")
    print(f"Service URL: {service.scoring_uri}")
    print(f"Service State: {service.state}")
    print(f"Deployment Target: {args.compute_target.upper()}")
    print("="*60 + "\n")
    
    logger.info("Deployment completed successfully!")


if __name__ == "__main__":
    main()

