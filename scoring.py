"""
scoring.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os
import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO
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
    """Run model inference."""
    # Convert raw data to a numpy array
    # data = np.array(json.loads(raw_data)['data'])

    # Load raw data
    data = json.loads(raw_data)

    # Get image arrays
    inputs = np.array([np.array(Image.open(BytesIO(base64.b64decode(row['image']))), dtype=np.uint8) for row in data])

    # Run model inverse with input data
    output = predictions.eval(session=sess, feed_dict={images: inputs})

    return json.dumps(output.tolist())
