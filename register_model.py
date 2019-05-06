"""
register_model.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.core import Workspace, Experiment, Run
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def register_model(args):
    """Upload MNIST dataset to Azure Workspace data store."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id,
                              resource_group=args.resource_group)

    # Get experiment
    experiment = Experiment(workspace=workspace, name=args.experiment_id)

    # Get run
    run = Run(experiment=experiment, run_id=args.run_id)

    # Register model
    model = run.register_model(model_name='mnist_tf_model', model_path='./outputs')
    print('\nModel Registration Complete:\nModel Name: {}\nModel ID: {}\nModel Version: {}\n'
          .format(model.name, model.id, model.version))


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument('--run_id', dest='run_id', type=str)
    parser.add_argument('--experiment_id', dest='experiment_id', type=str)
    parser.add_argument('--subscription_id', dest='subscription_id', type=str)
    parser.add_argument('--resource_group', dest='resource_group', type=str)

    return parser


if __name__ == '__main__':

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    register_model(args=arguments)
