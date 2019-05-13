"""
manage.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

from azureml.core import Workspace, Experiment, Run
from azureml.core.model import Model
from azureml.core.image import Image, ContainerImage


# Local imports
from mnistazure.config import DATA_PATH


def upload_data(args):
    """Upload MNIST dataset to Azure Workspace data store."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id,
                              resource_group=args.resource_group)

    # Get data store
    data_store = Datastore.get(workspace, datastore_name='workspacefilestore')

    # Upload MNIST dataset to data store
    data_store.upload(src_dir=DATA_PATH, target_path=None, show_progress=True)


def register_model(run_id, experiment_name, subscription_id, resource_group):
    """Upload MNIST dataset to Azure Workspace data store."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=subscription_id,
                              resource_group=resource_group)

    # Get experiment
    experiment = Experiment(workspace=workspace, name=experiment_name)

    # Get run
    run = Run(experiment=experiment, run_id=run_id)

    # Register model
    model = run.register_model(model_name='mnist_tf_model', model_path='./outputs')
    print('\nModel Registration Complete:\nModel Name: {}\nModel ID: {}\nModel Version: {}\n'
          .format(model.name, model.id, model.version))


def create_image_from_model(image_name, model_name, model_id, subscription_id, resource_group):
    """Create Docker image from registered model"""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=subscription_id,
                              resource_group=resource_group)

    # Get registered model
    model = Model(workspace=workspace, name=model_name, version=model_id)

    # Configure image
    image_config = ContainerImage.image_configuration(runtime='python', execution_script='scoring.py',
                                                      conda_file='service_env.yml')

    # Create image
    image = Image.create(name=image_name, models=[model], image_config=image_config, workspace=workspace)
    image.wait_for_creation(show_output=True)

