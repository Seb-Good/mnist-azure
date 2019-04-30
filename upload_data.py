"""
upload_data.py
--------------
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.core import Workspace
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Local imports
from mnistazure.config import DATA_PATH


def main(args):
    """Upload MNIST dataset to Azure Workspace data store."""
    # Get workspace
    ws = Workspace(subscription_id=args.subscription_id, resource_group=args.resource_group,
                   workspace_name=args.workspace_name)

    # Get data store
    ds = ws.get_default_datastore()

    # Upload MNIST dataset to data store
    ds.upload(src_dir=DATA_PATH, target_path='mnist', show_progress=True)


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument("--subscription_id", dest="subscription_id", type=str)
    parser.add_argument("--resource_group", dest="resource_group", type=str)
    parser.add_argument("--workspace_name", dest="workspace_name", type=str)

    return parser


if __name__ == "__main__":

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    main(args=arguments)
