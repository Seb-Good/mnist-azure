"""
train_azureml_local.py
----------------------
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.train.dnn import TensorFlow
from azureml.core import Workspace, Datastore, Experiment, RunConfiguration
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def train(args):
    """Train MNIST tensorflow model."""
    # Set input parameters
    script_params = {'--data_dir': r'C:\Users\sebig\Documents\Code\mnist-azure\data',
                     '--log_dir': './logs',
                     '--batch_size': 32,
                     '--learning_rate': 1e-3,
                     '--epochs': 5,
                     '--max_to_keep': 1,
                     '--seed': 0}

    # create local compute target
    run_local = RunConfiguration()
    run_local.environment.python.user_managed_dependencies = True

    # Get workspace
    ws = Workspace(subscription_id=args.subscription_id, resource_group=args.resource_group,
                   workspace_name='mnist-azure')

    print(ws.name)
    print(ws.get_details())
    print(ws.experiments)
    print('')
    for ct in ws.compute_targets:
        print(ct.name, ct.type)

    print(ws.get_default_compute_target(type='GPU'))

    # Get data store
    ds = Datastore.get(ws, datastore_name='workspacefilestore')

    # Define experiment
    ex = Experiment(workspace=ws, name='Experiment_2')

    tf_estimator = TensorFlow(source_directory='./', compute_target='local',
                              entry_script='train.py', script_params=script_params)

    run = ex.submit(tf_estimator)
    print(run.get_details())


def get_parser():
    """Get parser object for script train.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument("--subscription_id", dest="subscription_id", type=str)
    parser.add_argument("--resource_group", dest="resource_group", type=str)

    return parser


if __name__ == "__main__":

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    train(args=arguments)
