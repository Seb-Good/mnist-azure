"""
scoring.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os
import json
import numpy as np
import tensorflow as tf
from azureml.core.model import Model


def init():
    """Initialization function for model deployment."""
    # Set global variables
    global images, predictions, sess

    # Get model path
    model_path = Model.get_model_path(model_name='mnist_tf_model', version=5)

    # Start session
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Import meta graph
    saver = tf.train.import_meta_graph(os.path.join(model_path, 'graphs',
                                                    'inference_graph_{}.meta'.format('array')))

    # Get graph
    graph = tf.get_default_graph()

    # Get input tensor
    images = graph.get_tensor_by_name(name='images:0')

    # Get output tensor
    predictions = graph.get_tensor_by_name(name='prediction:0')

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Restore graph variables from checkpoint
    saver.restore(sess=sess, save_path=os.path.join(model_path, 'checkpoints', 'model'))


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    out = predictions.eval(session=sess, feed_dict={'images': data})
    y_hat = np.argmax(out, axis=1)
    return json.dumps(y_hat.tolist())
