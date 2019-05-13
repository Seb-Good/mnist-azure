"""
deploy_service_from_image.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.core import Workspace
from azureml.core.image import Image
from azureml.core.webservice import Webservice, AciWebservice
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def deploy_service(args):
    """Deploy service from Docker Image."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id,
                              resource_group=args.resource_group)

    # Get registered image
    image = Image(workspace=workspace, name=args.image_name, version=args.image_id)

    # Get webservice configuration
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=5,
                                                    description='Predict digit value from images '
                                                                'of handwritten digits')

    # Deploy service from Docker Image
    service = Webservice.deploy_from_image(workspace=workspace, name=args.service_name,
                                           deployment_config=aci_config, image=image)
    service.wait_for_deployment(show_output=True)


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument('--service_name', dest='service_name', type=str)
    parser.add_argument('--image_name', dest='image_name', type=str)
    parser.add_argument('--image_id', dest='image_id', type=int)
    parser.add_argument('--subscription_id', dest='subscription_id', type=str)
    parser.add_argument('--resource_group', dest='resource_group', type=str)

    return parser


if __name__ == '__main__':

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    deploy_service(args=arguments)
