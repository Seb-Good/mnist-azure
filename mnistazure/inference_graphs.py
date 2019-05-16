"""
inference_graph.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os
import tensorflow as tf


class InferenceGraph(object):

    def __init__(self, network):

        # Set parameters
        self.network = network

    def save(self, input_type):
        """Save inference graph for input type (string or array)."""
        # Start session
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # Get input graph
        images = self._get_input_graph(input_type=input_type)

        # Run inference and compute logits
        logits, _ = self.network.inference(input_layer=images, is_training=False)

        # Compute softmax score
        _ = tf.nn.softmax(logits=logits, axis=1, name='prediction')

        # Initialize global variables
        sess.run(tf.global_variables_initializer())

        # Save inference meta graph
        os.makedirs('./outputs/graphs/', exist_ok=True)
        tf.train.export_meta_graph('./outputs/graphs/inference_graph_{}.meta'.format(input_type))

        # Close session
        sess.close()

    def _get_input_graph(self, input_type):
        """Get string or array input."""
        if input_type is 'string':

            # Create input placeholder
            image_strings = tf.placeholder(tf.string, shape=[None], name='images')

            # Decode JPG image
            images = tf.map_fn(lambda val: tf.image.decode_jpeg(contents=val, channels=1),
                               elems=image_strings, dtype=tf.uint8)

            # Normalize RGB values between 0 and 1
            images = tf.map_fn(lambda val: tf.image.convert_image_dtype(image=val, dtype=tf.float32),
                               elems=images, dtype=tf.float32)

            # Get target height
            height = tf.cast(tf.round(tf.cast(tf.shape(images)[-3], dtype=tf.float32) /
                                      tf.cast((tf.shape(images)[-2] / self.network.width), dtype=tf.float32)),
                             dtype=tf.int32)

            # Resize with aspect ratio preserved
            images = tf.image.resize_images(images=images, size=[height, self.network.width])

            # Crop or Pad height
            images = tf.image.resize_image_with_crop_or_pad(images, self.network.height, self.network.width)

            # Set tensor shape
            images = tf.reshape(tensor=images, shape=[-1, self.network.height, self.network.width, 1])

            return images

        elif input_type is 'array':

            # Get image array placeholder
            images, _ = self.network.create_placeholders()

            return images
