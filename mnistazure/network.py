"""
network.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import tensorflow as tf

# Local imports
from mnistazure.generator import DataGenerator


class Network(object):

    def __init__(self, dropout_rate, height, width, channels, num_labels, seed=0):

        # Set input parameters
        self.dropout_rate = dropout_rate
        self.height = height
        self.width = width
        self.channels = channels
        self.num_labels = num_labels
        self.seed = seed

    def inference(self, input_layer, is_training):

        """Forward propagation of computational graph."""

        # Check input dimensions
        assert input_layer.shape[1] == self.height
        assert input_layer.shape[2] == self.width
        assert input_layer.shape[3] == self.channels

        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet'):

            # Conv layer 1
            net = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5],
                                   padding='SAME', activation=tf.nn.relu)

            # Max pool 1
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

            # Conv layer 2
            net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5], padding='SAME', activation=tf.nn.relu)

            # Max pool 2
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

            # Flatten
            net = tf.layers.flatten(net)

            # Dense layer
            net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)

            # Dropout layer
            net = tf.layers.dropout(inputs=net, rate=self.dropout_rate, training=is_training)

            # Logits layer
            logits = tf.layers.dense(inputs=net, units=10)

            # Predictions
            predictions = {'classes': tf.argmax(input=logits, axis=1),
                           'probabilities': tf.nn.softmax(logits, name="softmax_tensor")}

            # Check output dimensions
            assert logits.shape[1] == self.num_labels

            return logits, predictions

    def create_placeholders(self):
        """Creates place holders for images and labels."""
        # Create placeholders
        images = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, self.channels], name='images')
        labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        return images, labels

    def create_generator(self, path, mode, batch_size):
        """Create data generator graph operation."""
        return DataGenerator(path=path, mode=mode, shape=[self.height, self.width, self.channels],
                             batch_size=batch_size, prefetch_buffer=100, seed=0, num_parallel_calls=6)

    @staticmethod
    def compute_accuracy(logits, labels):
        """Computes the accuracy for a given set of predicted logits and true labels."""
        # Cast labels to int64
        labels = tf.cast(labels, tf.int64)

        # Compute set accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), 'float'))

        return accuracy
