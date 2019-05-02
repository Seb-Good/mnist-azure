"""
train.py
--------
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Local imports
from azureml.core import Run
from mnistazure.graph import Graph
from mnistazure.network import Network


def train(args):
    """Train MNIST tensorflow model."""

    # Get run context
    run = Run.get_context()

    # Image shape
    image_shape = (28, 28, 1)

    # Number of unique labels
    num_labels = 10

    # Initialize network
    network = Network(height=image_shape[0], width=image_shape[1],
                      channels=image_shape[2], num_labels=num_labels, seed=0)

    # Initialize graph
    graph = Graph(network=network, save_path=args.log_dir, data_path=args.data_dir, max_to_keep=args.max_to_keep)

    with tf.Session() as sess:

        # Initialize variables
        sess.run(graph.init_global)

        # Summary writer
        train_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'train'), sess.graph)

        # Get number of training batches
        num_train_batches = graph.generator_train.num_batches.eval(
            feed_dict={graph.batch_size: args.batch_size})

        # Get number of batch steps per epoch
        steps_per_epoch = int(np.ceil(num_train_batches / 1))

        # Get mode handle for training
        handle_train = sess.run(graph.generator_train.iterator.string_handle())

        # Initialize the train dataset iterator at the beginning of each epoch
        sess.run(fetches=[graph.generator_train.iterator.initializer],
                 feed_dict={graph.batch_size: args.batch_size})

        # Loop through epochs
        for epoch in range(args.epochs):

            # Initialize metrics
            sess.run(fetches=[graph.init_metrics_op])

            # Loop through train dataset batches
            for batch in range(steps_per_epoch):

                # Run train operation
                loss, accuracy, _, _, summaries, global_step = sess.run(
                    fetches=[graph.loss, graph.accuracy, graph.train_op, graph.update_metrics_op,
                             graph.train_summary_metrics_op, graph.global_step],
                    feed_dict={graph.batch_size: args.batch_size, graph.is_training: True,
                               graph.learning_rate: args.learning_rate, graph.mode_handle: handle_train}
                )

                # Print performance
                if batch % 10 == 0:
                    print('Loss: {}, Accuracy: {}'.format(loss, accuracy))
                    if run is not None:
                        run.log('loss', loss)
                        run.log('accuracy', accuracy)

                # Collect summaries
                train_writer.add_summary(summary=summaries, global_step=global_step)

            # Initialize the train dataset iterator at the end of each epoch
            sess.run(fetches=[graph.generator_train.iterator.initializer],
                     feed_dict={graph.batch_size: args.batch_size})

        # Save checkpoint
        os.makedirs('./outputs/checkpoints/', exist_ok=True)
        graph.saver.save(sess=sess, save_path='./outputs/checkpoints/model')
        print('Checkpoint Saved')

        # Close summary writer
        train_writer.close()

    # Complete logging
    run.complete()


def get_parser():
    """Get parser object for script train.py."""
    # Initialize parser
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)

    # Setup arguments
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--log_dir", dest="log_dir", type=str)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", dest="epochs", type=int, default=5)
    parser.add_argument("--max_to_keep", dest="max_to_keep", type=int, default=1)
    parser.add_argument("--seed", dest="seed", type=int, default=0)

    return parser


if __name__ == "__main__":

    # Parse arguments
    arguments = get_parser().parse_args()

    # Run main function
    train(args=arguments)
