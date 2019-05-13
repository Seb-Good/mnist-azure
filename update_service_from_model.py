"""
update_service_from_model.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image, ContainerImage
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def update_service(args):
    """Update service from model."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id,
                              resource_group=args.resource_group)

    # Get registered model
    model = Model(workspace=workspace, name=args.model_name, version=args.model_id)

    # Configure image
    image_config = ContainerImage.image_configuration(runtime='python', execution_script='scoring.py',
                                                      conda_file='service_env.yml')

    # Create image
    image = Image.create(name=args.service_name, models=[model], image_config=image_config, workspace=workspace)
    image.wait_for_creation(show_output=True)

    # Get web service
    service = workspace.webservices[args.service_name]

    # Update service
    service.update(image=image)
    service.wait_for_deployment(show_output=True)


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument('--service_name', dest='service_name', type=str)
    parser.add_argument('--model_name', dest='model_name', type=str)
    parser.add_argument('--model_id', dest='model_id', type=int)
    parser.add_argument('--subscription_id', dest='subscription_id', type=str)
    parser.add_argument('--resource_group', dest='resource_group', type=str)

    return parser


if __name__ == '__main__':

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    update_service(args=arguments)
