"""
azureml.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
from azureml.train.estimator import Estimator
from azureml.core import Workspace, Experiment, Run, RunConfiguration


def run_experiment(subscription_id, resource_group, experiment_id, compute_target,
                   environment_definition, script_params):
    """Run experiment to train MNIST tensorflow model."""
    # Get workspace
    workspace = Workspace.get(name='mnist-azure', subscription_id=subscription_id,
                              resource_group=resource_group)

    # Define experiment
    experiment = Experiment(workspace=workspace, name=experiment_id)

    # Define estimator
    estimator = Estimator(source_directory='./', compute_target=compute_target, entry_script='train.py',
                          script_params=script_params, environment_definition=environment_definition)

    # Run experiment
    run = experiment.submit(estimator)
    run.wait_for_completion(show_output=True)
