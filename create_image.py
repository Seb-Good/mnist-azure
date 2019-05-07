"""
create_image.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import Image, ContainerImage
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def create_image(args):
    """Create Docker image from registered model"""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id,
                              resource_group=args.resource_group)

    # Get registered model
    model = Model(workspace=workspace, name=args.model_name, version=args.model_id)

    # Configure image
    image_config = ContainerImage.image_configuration(runtime='python', execution_script='scoring.py',
                                                      conda_file='environment.yml')

    # Create image
    _ = Image.create(name=args.image_name, models=[model], image_config=image_config, workspace=workspace)


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument('--image_name', dest='image_name', type=str)
    parser.add_argument('--model_name', dest='model_name', type=str)
    parser.add_argument('--model_id', dest='model_id', type=int)
    parser.add_argument('--subscription_id', dest='subscription_id', type=str)
    parser.add_argument('--resource_group', dest='resource_group', type=str)

    return parser


if __name__ == '__main__':

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    create_image(args=arguments)

