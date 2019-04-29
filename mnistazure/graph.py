"""
graph.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import tensorflow as tf


class Graph(object):

    def __init__(self, network, save_path, data_path, max_to_keep):

        # Set input parameters
        self.network = network
        self.save_path = save_path
        self.data_path = data_path
        self.max_to_keep = max_to_keep

        # Set attributes
        self.is_training = None
        self.learning_rate = None
        self.global_step = None
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.train_op = None
        self.train_summary_metrics_op = None
        self.init_global = None
        self.saver = None
        self.generator_train = None
        self.generator_val = None
        self.batch_size = None
        self.num_train_batches = None
        self.num_val_batches = None
        self.mode_handle = None
        self.images = None
        self.labels = None
        self.iterator = None
        self.metrics = None
        self.update_metrics_op = None
        self.init_metrics_op = None
        self.gradients = None

        # Build computational graph
        self.build_graph()

    def build_graph(self):
        """Constructs a computational graph for training and evaluating a deep neural network."""
        """Setup"""
        # Reset graph
        tf.reset_default_graph()

        """Initialize Variables and Placeholders"""
        # Create placeholders
        self.is_training, self.learning_rate, self.batch_size, self.mode_handle = self._create_placeholders()

        # Get or create global step
        self.global_step = tf.train.get_or_create_global_step()

        """Data Generators"""
        # Data train, val, and test data generators
        self.generator_train, self.generator_val = self._get_generators()

        # Initialize iterator
        self.iterator = self._initialize_iterator()

        # Get batch
        self.images, self.labels = self._get_next_batch()

        """Compute Tower Gradients"""
        # Initialize optimizer
        self.optimizer = self._create_optimizer(learning_rate=self.learning_rate)

        # Compute inference
        self.logits, _ = self.network.inference(input_layer=self.images, is_training=self.is_training)

        # Compute loss
        self.loss = self._compute_loss(logits=self.logits, labels=self.labels)

        # Compute accuracy
        self.accuracy = self._compute_accuracy(logits=self.logits, labels=self.labels)

        # Compute gradients
        self.gradients = self._compute_gradients(optimizer=self.optimizer, loss=self.loss)

        """Run Optimization"""
        # Training step
        self.train_op = self._run_optimization(grads_and_vars=self.gradients, global_step=self.global_step)

        """Metrics"""
        # Compute metrics
        self.metrics = self._compute_metrics()

        # Run metrics update
        self.update_metrics_op = self._update_metrics()

        # Initialize metrics
        self.init_metrics_op = self._initialize_metrics()

        """Summaries"""
        # Merge training summaries
        self.train_summary_metrics_op = tf.summary.merge_all(key='train_metrics')

        """Initialize Variables"""
        # Initialize global variables
        self.init_global = tf.global_variables_initializer()

        """Save Checkpoints"""
        # Initialize saver
        self.saver = self._get_saver()

    """Methods: Metrics"""
    def _compute_metrics(self):
        """Metrics for evaluation using tf.metrics (average over complete dataset)."""
        with tf.variable_scope('metrics'):
            metrics = {'accuracy': tf.metrics.mean(values=self.accuracy),
                       'loss': tf.metrics.mean(values=self.loss)}
        return metrics

    def _update_metrics(self):
        """Group metric update ops."""
        return tf.group(*[op for _, op in self.metrics.values()])

    @staticmethod
    def _compute_loss(logits, labels):
        """Computes the model cross-entropy loss for a given set of predicted logits and true labels."""
        with tf.variable_scope('loss'):
            # Cast labels as int32
            labels = tf.cast(labels, tf.int32)

            # Compute cross entropy losses from logits of final layer
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Compute mean loss
            loss = tf.reduce_mean(losses)

        # Get summary
        tf.summary.scalar(name='loss/loss', tensor=loss, collections=['train_metrics'])

        return loss

    def _compute_accuracy(self, logits, labels):
        """Computes the accuracy for a given set of predicted logits and true labels."""
        with tf.variable_scope('accuracy'):

            # Compute accuracy
            accuracy = self.network.compute_accuracy(logits=logits, labels=labels)

        # Get summary
        tf.summary.scalar(name='accuracy/accuracy', tensor=accuracy, collections=['train_all', 'train_metrics'])

        return accuracy

    @staticmethod
    def _initialize_metrics():
        """Initialize metrics."""
        metric_variables = tf.get_collection(key=tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        return tf.variables_initializer(var_list=metric_variables)

    """Methods: Optimization"""
    @staticmethod
    def _compute_gradients(optimizer, loss):
        """Computes the model gradients for a given cross-entropy loss."""
        gradients = optimizer.compute_gradients(loss=loss)
        return gradients

    @staticmethod
    def _create_optimizer(learning_rate):
        """Create an optimizer using the ATOM algorithm."""
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

        # Get learning rate summary
        tf.summary.scalar(name='learning_rate/learning_rate', tensor=learning_rate, collections=['train_metrics'])

        return optimizer

    def _run_optimization(self, grads_and_vars, global_step):
        """Runs objective function optimization for one training step."""
        with tf.variable_scope('train_op'):
            # Run optimization step
            train_op = self.optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        return train_op

    """Methods: Setup"""
    @staticmethod
    def _create_placeholders():
        """Creates place holders for neural network."""
        with tf.variable_scope('is_training') as scope:
            is_training = tf.placeholder_with_default(True, shape=(), name=scope.name)

        with tf.variable_scope('learning_rate') as scope:
            learning_rate = tf.placeholder(dtype=tf.float32, name=scope.name)

        with tf.variable_scope('batch_size') as scope:
            batch_size = tf.placeholder(dtype=tf.int64, name=scope.name)

        with tf.variable_scope('mode_handle') as scope:
            mode_handle = tf.placeholder(dtype=tf.string, shape=[], name=scope.name)

        return is_training, learning_rate, batch_size, mode_handle

    """Methods: Data Generator"""
    def _initialize_iterator(self):
        """Initialize the iterator from a mode handle placeholder"""
        with tf.variable_scope('iterator'):

            # Get iterator
            iterator = tf.data.Iterator.from_string_handle(self.mode_handle, self.generator_train.dataset.output_types,
                                                           self.generator_train.dataset.output_shapes)
        return iterator

    def _get_next_batch(self):
        """Get next batch (images, labels) from iterator."""
        with tf.name_scope('next_batch'):
            images, labels = self.iterator.get_next()

        return images, labels

    def _get_generators(self):
        """Create train and val data generators."""
        with tf.variable_scope('train_generator'):
            generator_train = self.network.create_generator(path=self.data_path, mode='train',
                                                            batch_size=self.batch_size)
        with tf.variable_scope('val_generator'):
            generator_val = self.network.create_generator(path=self.data_path, mode='val',
                                                          batch_size=self.batch_size)
        return generator_train, generator_val

    def _get_saver(self):
        """Create tensorflow checkpoint saver"""
        with tf.variable_scope('saver'):
            return tf.train.Saver(max_to_keep=self.max_to_keep)
