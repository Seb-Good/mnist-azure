"""
train_azureml_local.py
----------------------
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.tensorboard import Tensorboard
from azureml.train.estimator import Estimator
from azureml.core import Workspace, Experiment, RunConfiguration
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Local imports
from mnistazure.config import DATA_PATH


def run_experiment(args):
    """Train MNIST tensorflow model."""
    # Set input parameters
    script_params = {'--data_dir': DATA_PATH,
                     '--log_dir': './logs',
                     '--batch_size': 32,
                     '--learning_rate': 1e-3,
                     '--epochs': 5,
                     '--max_to_keep': 1,
                     '--seed': 0}

    # Get local run configuration
    run_local = RunConfiguration()
    run_local.environment.python.user_managed_dependencies = True

    # Get workspace
    ws = Workspace.get(name='mnist-azure', subscription_id=args.subscription_id, resource_group=args.resource_group)

    # Define experiment
    exp = Experiment(workspace=ws, name=args.experiment_name)

    # Define estimator
    estimator = Estimator(source_directory='./', compute_target='local', entry_script='train.py',
                          script_params=script_params, environment_definition=run_local.environment)

    # Run experiment
    run = exp.submit(estimator)
    run.wait_for_completion(show_output=True)

    # Print tensorboard url
    tb = Tensorboard(run)
    tb.start()


def get_parser():
    """Get parser object for script train.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument("--subscription_id", dest="subscription_id", type=str)
    parser.add_argument("--resource_group", dest="resource_group", type=str)
    parser.add_argument("--experiment_name", dest="experiment_name", type=str)

    return parser


if __name__ == "__main__":

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    run_experiment(args=arguments)
