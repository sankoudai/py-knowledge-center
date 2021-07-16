import unittest

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from cmp_util import *

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


class TestIntegerPreprocess(unittest.TestCase):
    def test_hashing(self):
        data = tf.constant([990, 1000], dtype=tf.int32)

        # Use the Hashing layer to hash the values to the range [0, 64]
        hasher = preprocessing.Hashing(num_bins=64, salt=1337)

        # Use the CategoryEncoding layer to one-hot encode the hashed values
        assert_equal(hasher(data), [13, 46])

    def test_index(self):
        '''
        IntegerLookup:
            0 : reserved  missing values
            1 : reserved for out-of-vocabulary values
        '''
        # Define some toy data
        data = tf.constant([10, 20, 20, 10, 30, 0])

        # Use IntegerLookup to build an index of the feature values
        indexer = preprocessing.IntegerLookup()
        indexer.adapt(data)

        # Use CategoryEncoding to encode the integer indices to a one-hot vector
        test_data = tf.constant([100, 10])
        print(indexer(test_data))

    def test_embedding(self):
        # Define some toy data
        data = tf.constant([10, 20, 20, 10, 30, 0])

        emb_layer = tf.keras.layers.Embedding(input_dim=60, output_dim=10)
        print(emb_layer(data))

    def test_onehot(self):
        '''
            CategoryEncoding： 一般需要先index或hash，然后进行onehot编码
        '''
        data = tf.constant([10, 20, 20, 10, 300, 0])

        hasher = preprocessing.Hashing(num_bins=64, salt=1337)
        encoder = preprocessing.CategoryEncoding(output_mode="binary", num_tokens=hasher.num_bins)
        print(encoder(hasher(data)))