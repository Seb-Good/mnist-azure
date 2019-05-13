"""
pipeline_1.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os
import numpy as np
import tensorflow as tf
from azureml.core import Run
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def download_data(args):
    """Download MNIST dataset and save to datastore."""
    # Get run context
    run = Run.get_context()
    if run is not None:
        run.log('start', 'start')

    # Create directory
    os.makedirs(args.output, exist_ok=True)
    if run is not None:
        run.log('output_dir', args.output)

    # Download MNIST
    ((train_data, train_labels), (val_data, val_labels)) = tf.keras.datasets.mnist.load_data()
    if run is not None:
        run.log('download', 'done')

    # Save as .npy files
    np.save(os.path.join(args.output, 'train_data.npy'), train_data)
    np.save(os.path.join(args.output, 'train_labels.npy'), train_labels)
    np.save(os.path.join(args.output, 'val_data.npy'), val_data)
    np.save(os.path.join(args.output, 'val_labels.npy'), val_labels)
    if run is not None:
        run.log('val_labels_path', os.path.join(args.output, 'val_labels.npy'))


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument('--output', dest='output', type=str)

    return parser


if __name__ == '__main__':

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    download_data(args=arguments)
