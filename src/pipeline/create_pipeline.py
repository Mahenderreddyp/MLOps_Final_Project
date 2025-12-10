"""
Azure ML Pipeline Creation Script

This script creates the Azure ML pipeline for hospital cost prediction.
Based on the pipeline components shown in the screenshots:
1. Data ingestion
2. Data cleaning
3. SQL transformations
4. Feature selection
5. Data splitting
6. Model training (Boosted Decision Tree)
7. Hyperparameter tuning
8. Feature importance
9. Model scoring
10. Metrics calculation
"""

import os
from azureml.core import Workspace, Dataset, Datastore, Environment, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.data.datapath import DataPath
from azureml.core.runconfig import RunConfiguration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HospitalCostPipeline:
    """Class to create and manage Azure ML pipeline"""
    
    def __init__(
        self,
        workspace: Workspace,
        compute_name: str = "cpu-cluster",
        experiment_name: str = "hospital-cost-prediction"
    ):
        """
        Initialize pipeline builder
        
        Args:
            workspace: Azure ML Workspace
            compute_name: Name of compute target
            experiment_name: Name of experiment
        """
        self.ws = workspace
        self.compute_name = compute_name
        self.experiment_name = experiment_name
        self.logger = logger
        
        # Get or create compute target
        self.compute_target = self._get_compute_target()
        
        # Create environment
        self.environment = self._create_environment()
        
        # Create run configuration
        self.run_config = self._create_run_config()
    
    def _get_compute_target(self) -> ComputeTarget:
        """Get or create compute target"""
        try:
            compute_target = ComputeTarget(workspace=self.ws, name=self.compute_name)
            self.logger.info(f"Found existing compute target: {self.compute_name}")
        except Exception:
            self.logger.info(f"Creating new compute target: {self.compute_name}")
            compute_config = AmlCompute.provisioning_configuration(
                vm_size='STANDARD_D3_V2',
                max_nodes=4,
                idle_seconds_before_scaledown=300
            )
            compute_target = ComputeTarget.create(self.ws, self.compute_name, compute_config)
            compute_target.wait_for_completion(show_output=True)
        
        return compute_target
    
    def _create_environment(self) -> Environment:
        """Create environment with required packages"""
        env = Environment(name="hospital-cost-env")
        
        # Python packages
        env.python.conda_dependencies.add_pip_package("pandas==2.0.3")
        env.python.conda_dependencies.add_pip_package("numpy==1.24.3")
        env.python.conda_dependencies.add_pip_package("scikit-learn==1.3.0")
        env.python.conda_dependencies.add_pip_package("joblib==1.3.1")
        env.python.conda_dependencies.add_pip_package("azureml-core")
        env.python.conda_dependencies.add_pip_package("azureml-dataset-runtime")
        
        return env
    
    def _create_run_config(self) -> RunConfiguration:
        """Create run configuration"""
        run_config = RunConfiguration()
        run_config.target = self.compute_target
        run_config.environment = self.environment
        return run_config
    
    def create_data_prep_step(
        self,
        input_dataset: Dataset,
        output_data: PipelineData
    ) -> PythonScriptStep:
        """
        Create data preparation step
        
        Args:
            input_dataset: Input dataset
            output_data: Output pipeline data
            
        Returns:
            PythonScriptStep for data preparation
        """
        return PythonScriptStep(
            name="data_preparation",
            script_name="preprocess.py",
            arguments=[
                "--input-data", input_dataset.as_named_input('raw_data'),
                "--output-data", output_data
            ],
            inputs=[input_dataset.as_named_input('raw_data')],
            outputs=[output_data],
            compute_target=self.compute_target,
            runconfig=self.run_config,
            source_directory="./src/data_processing",
            allow_reuse=True
        )
    
    def create_training_step(
        self,
        input_data: PipelineData,
        model_output: PipelineData
    ) -> PythonScriptStep:
        """
        Create model training step
        
        Args:
            input_data: Input pipeline data
            model_output: Output for trained model
            
        Returns:
            PythonScriptStep for training
        """
        return PythonScriptStep(
            name="train_model",
            script_name="train.py",
            arguments=[
                "--train-data", input_data,
                "--output-model", model_output
            ],
            inputs=[input_data],
            outputs=[model_output],
            compute_target=self.compute_target,
            runconfig=self.run_config,
            source_directory="./src/models",
            allow_reuse=True
        )
    
    def create_evaluation_step(
        self,
        model_data: PipelineData,
        test_data: PipelineData,
        metrics_output: PipelineData
    ) -> PythonScriptStep:
        """
        Create model evaluation step
        
        Args:
            model_data: Trained model
            test_data: Test dataset
            metrics_output: Output for metrics
            
        Returns:
            PythonScriptStep for evaluation
        """
        return PythonScriptStep(
            name="evaluate_model",
            script_name="evaluate.py",
            arguments=[
                "--model", model_data,
                "--test-data", test_data,
                "--metrics-output", metrics_output
            ],
            inputs=[model_data, test_data],
            outputs=[metrics_output],
            compute_target=self.compute_target,
            runconfig=self.run_config,
            source_directory="./src/models",
            allow_reuse=True
        )
    
    def build_pipeline(self, input_dataset: Dataset) -> Pipeline:
        """
        Build the complete pipeline
        
        Args:
            input_dataset: Input dataset
            
        Returns:
            Complete Azure ML Pipeline
        """
        self.logger.info("Building Azure ML Pipeline...")
        
        # Create pipeline data objects
        datastore = self.ws.get_default_datastore()
        
        processed_data = PipelineData(
            "processed_data",
            datastore=datastore,
            output_mode="mount"
        )
        
        model_data = PipelineData(
            "model_data",
            datastore=datastore,
            output_mode="mount"
        )
        
        metrics_data = PipelineData(
            "metrics_data",
            datastore=datastore,
            output_mode="mount"
        )
        
        # Create pipeline steps
        data_prep_step = self.create_data_prep_step(input_dataset, processed_data)
        training_step = self.create_training_step(processed_data, model_data)
        evaluation_step = self.create_evaluation_step(model_data, processed_data, metrics_data)
        
        # Build pipeline
        pipeline = Pipeline(
            workspace=self.ws,
            steps=[data_prep_step, training_step, evaluation_step],
            description="Hospital Cost Prediction Pipeline"
        )
        
        self.logger.info("Pipeline built successfully!")
        return pipeline
    
    def publish_pipeline(self, pipeline: Pipeline, pipeline_name: str) -> None:
        """
        Publish the pipeline
        
        Args:
            pipeline: Pipeline to publish
            pipeline_name: Name for published pipeline
        """
        self.logger.info(f"Publishing pipeline as: {pipeline_name}")
        
        published_pipeline = pipeline.publish(
            name=pipeline_name,
            description="Published Hospital Cost Prediction Pipeline",
            version="1.0"
        )
        
        self.logger.info(f"Pipeline published with ID: {published_pipeline.id}")
        return published_pipeline
    
    def run_pipeline(self, pipeline: Pipeline) -> None:
        """
        Run the pipeline
        
        Args:
            pipeline: Pipeline to run
        """
        self.logger.info(f"Submitting pipeline to experiment: {self.experiment_name}")
        
        # Create experiment
        experiment = Experiment(workspace=self.ws, name=self.experiment_name)
        
        # Submit pipeline
        pipeline_run = experiment.submit(pipeline)
        
        self.logger.info(f"Pipeline submitted. Run ID: {pipeline_run.id}")
        self.logger.info("Waiting for pipeline to complete...")
        
        # Wait for completion
        pipeline_run.wait_for_completion(show_output=True)
        
        self.logger.info("Pipeline completed!")


def main():
    """Main function to create and run pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Azure ML Pipeline')
    
    parser.add_argument(
        '--subscription-id',
        type=str,
        required=True,
        help='Azure subscription ID'
    )
    
    parser.add_argument(
        '--resource-group',
        type=str,
        required=True,
        help='Azure resource group'
    )
    
    parser.add_argument(
        '--workspace-name',
        type=str,
        required=True,
        help='Azure ML workspace name'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='sparcs_raw',
        help='Name of registered dataset'
    )
    
    parser.add_argument(
        '--run',
        action='store_true',
        help='Run the pipeline after creation'
    )
    
    parser.add_argument(
        '--publish',
        action='store_true',
        help='Publish the pipeline'
    )
    
    args = parser.parse_args()
    
    # Connect to workspace
    logger.info("Connecting to Azure ML Workspace...")
    ws = Workspace(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    logger.info(f"Connected to workspace: {ws.name}")
    
    # Get dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = Dataset.get_by_name(ws, args.dataset_name)
    
    # Create pipeline
    pipeline_builder = HospitalCostPipeline(ws)
    pipeline = pipeline_builder.build_pipeline(dataset)
    
    # Publish if requested
    if args.publish:
        pipeline_builder.publish_pipeline(pipeline, "hospital-cost-prediction-pipeline")
    
    # Run if requested
    if args.run:
        pipeline_builder.run_pipeline(pipeline)
    
    logger.info("Pipeline creation completed!")


if __name__ == "__main__":
    main()
