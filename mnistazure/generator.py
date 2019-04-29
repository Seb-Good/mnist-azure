"""
generator.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os
import json
import tensorflow as tf


class DataGenerator(object):

    def __init__(self, path, mode, shape, batch_size, prefetch_buffer=1, seed=0, num_parallel_calls=1):

        # Set parameters
        self.path = path
        self.mode = mode
        self.shape = shape
        self.batch_size = batch_size
        self.prefetch_buffer = prefetch_buffer
        self.seed = seed
        self.num_parallel_calls = num_parallel_calls

        # Set attributes
        self.lookup_dict = self._get_lookup_dict()
        self.file_names = self._get_file_names()
        self.labels = self._get_labels()
        self.num_samples = len(self.labels)
        self.file_paths = self._get_file_paths()
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.current_seed = 0

        # Get lambda functions
        self.import_image_train_fn = lambda file_path, label: self._import_image(file_path=file_path, label=label)
        self.import_image_val_fn = lambda file_path, label: self._import_image(file_path=file_path, label=label)

        # Get dataset
        self.dataset = self._get_dataset()

        # Get iterator
        self.iterator = self.dataset.make_initializable_iterator()

    def _get_next_seed(self):
        """update current seed"""
        self.current_seed += 1
        return self.current_seed

    def _get_lookup_dict(self):
        """Load lookup dictionary {'file_name': label}."""
        return json.load(open(os.path.join(self.path, 'labels', 'labels.json')))

    def _get_file_names(self):
        """Get list of image file names."""
        return [val[0] for val in self.lookup_dict[self.mode]]

    def _get_labels(self):
        """Get list of labels."""
        return [val[1] for val in self.lookup_dict[self.mode]]

    def _get_file_paths(self):
        """Convert file names to full absolute file paths with .jpg extension."""
        return [os.path.join(self.path, 'images', '{}.jpg'.format(file_name)) for file_name in self.file_names]

    def _import_image(self, file_path, label):
        """Import and decode image files from file path strings."""
        # Get image file name as string
        image_string = tf.read_file(filename=file_path)

        # Decode JPG image
        image_decoded = tf.image.decode_jpeg(contents=image_string, channels=3)

        # Normalize RGB values between 0 and 1
        image_normalized = tf.image.convert_image_dtype(image=image_decoded, dtype=tf.float32)

        # Set tensor shape
        image = tf.reshape(tensor=image_normalized, shape=self.shape)

        return image, label

    def _get_dataset(self):
        """Retrieve tensorflow Dataset object."""
        if self.mode == 'train':
            return (
                tf.data.Dataset.from_tensor_slices(tensors=(tf.constant(value=self.file_paths),
                                                            tf.reshape(tensor=tf.constant(self.labels), shape=[-1])))
                .shuffle(buffer_size=self.num_samples, reshuffle_each_iteration=True)
                .map(map_func=self.import_image_train_fn, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )
        else:
            return (
                tf.data.Dataset.from_tensor_slices(tensors=(tf.constant(value=self.file_paths),
                                                            tf.reshape(tensor=tf.constant(self.labels), shape=[-1])))
                .map(map_func=self.import_image_val_fn, num_parallel_calls=self.num_parallel_calls)
                .repeat()
                .batch(batch_size=self.batch_size)
                .prefetch(buffer_size=self.prefetch_buffer)
            )

    def _import_images(self, file_path, label):
        """Import and decode image files from file path strings."""
        # Get image file name as string
        image_string = tf.read_file(filename=file_path)

        # Decode JPG image
        image_decoded = tf.image.decode_jpeg(contents=image_string, channels=3)

        # Normalize RGB values between 0 and 1
        image_normalized = tf.image.convert_image_dtype(image=image_decoded, dtype=tf.float32)

        # Set tensor shape
        image = tf.reshape(tensor=image_normalized, shape=self.shape)

        return image, label
