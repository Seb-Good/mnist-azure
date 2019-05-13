"""
deploy_service_from_image.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.image import ContainerImage
from azureml.core.webservice import Webservice, AciWebservice
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def deploy_service(args):
    """Deploy service from model."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id,
                              resource_group=args.resource_group)

    # Get registered model
    model = Model(workspace=workspace, name=args.model_name, version=args.model_id)

    # Configure image
    image_config = ContainerImage.image_configuration(runtime='python', execution_script='scoring.py',
                                                      conda_file='service_env.yml')

    # Get webservice configuration
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=5,
                                                    description='Predict digit value from images '
                                                                'of handwritten digits')

    # Deploy service from Docker Image
    service = Webservice.deploy_from_model(workspace=workspace, name=args.service_name, deployment_config=aci_config,
                                           image_config=image_config, models=[model])
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
    deploy_service(args=arguments)
