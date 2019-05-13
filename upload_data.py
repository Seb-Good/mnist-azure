"""
upload_data.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.core import Workspace, Datastore
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Local imports
from mnistazure.config import DATA_PATH


def upload_data(args):
    """Upload MNIST dataset to Azure Workspace data store."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id,
                              resource_group=args.resource_group)

    # Get data store
    data_store = Datastore.get(workspace=workspace, datastore_name='workspacefilestore')

    # Upload MNIST dataset to data store
    data_store.upload(src_dir=DATA_PATH, target_path=None, show_progress=True)


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument('--subscription_id', dest='subscription_id', type=str)
    parser.add_argument('--resource_group', dest='resource_group', type=str)

    return parser


if __name__ == '__main__':

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    upload_data(args=arguments)
