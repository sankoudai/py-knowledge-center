import tensorflow as tf
import numpy as np
import unittest
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


class RecipeOnehotEncoding(unittest.TestCase):
    def test_onehot(self):
        # Define some toy data
        data = tf.constant(["a", "b", "c", "d", "c", "a"])

        # Use StringLookup to build an index of the feature values
        indexer = preprocessing.StringLookup()
        indexer.adapt(data)

        # Use CategoryEncoding to encode the integer indices to a one-hot vector
        encoder = preprocessing.CategoryEncoding(output_mode="binary", num_tokens=indexer.vocab_size())
        encoder.adapt(indexer(data))

        # Convert new test data (which includes unknown feature values)
        test_data = tf.constant(["a", "b", "c", "d", "e", ""])
        encoded_data = encoder(indexer(test_data))
        print(encoded_data)