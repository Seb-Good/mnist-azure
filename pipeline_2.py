"""
pipeline_1.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os
import cv2
import json
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def create_training_data(args):
    """Create training dataset and save to datastore."""
    # Import data
    train_data = np.load(os.path.join(args.input, 'train_data.npy'))
    train_labels = np.load(os.path.join(args.input, 'train_labels.npy'))
    val_data = np.load(os.path.join(args.input, 'val_data.npy'))
    val_labels = np.load(os.path.join(args.input, 'val_labels.npy'))
    print('Data finished loading.')

    # Create folders
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'labels'), exist_ok=True)
    print('Folders created.')

    # Labels dictionary
    labels = {'train': [], 'val': []}

    # Generate train dataset
    for idx in range(10):
        # Set file name
        file_name = 'train_{}'.format(idx)

        # Get label
        labels['train'].append((file_name, int(train_labels[idx])))

        # Save jpeg
        cv2.imwrite(os.path.join(args.output, 'images', '{}.jpg'.format(file_name)), train_data[idx])

    # Generate val dataset
    for idx in range(10):
        # Set file name
        file_name = 'val_{}'.format(idx)

        # Get label
        labels['val'].append((file_name, int(val_labels[idx])))

        # Save jpeg
        cv2.imwrite(os.path.join(args.output, 'images', '{}.jpg'.format(file_name)), val_data[idx])

    # Save labels
    with open(os.path.join(args.output, 'labels', 'labels.json'), 'w') as file:
        json.dump(labels, file, sort_keys=True)


def get_parser():
    """Get parser object for script upload_data.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument('--input', dest='input', type=str)
    parser.add_argument('--output', dest='output', type=str)

    return parser


if __name__ == '__main__':

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    create_training_data(args=arguments)
