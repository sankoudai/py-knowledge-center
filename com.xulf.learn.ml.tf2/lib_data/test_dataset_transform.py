import unittest
import tensorflow as tf
import numpy as np
from dataset_util import first_element_of
from cmp_util import *

np.set_printoptions(precision=3)


class TestDatasetTransform(unittest.TestCase):
    def test_map(self):
        """
            ds.map(func):
            - func: use tf ops whenever possible; use tf.py_function if non-tf ops is needed
        """

        elems = [(1, "foo"), (2, "bar"), (3, "baz")]
        ds = tf.data.Dataset.from_generator(lambda: elems,
                                            output_signature=(tf.TensorSpec(shape=(), dtype=tf.int32),
                                                              tf.TensorSpec(shape=(), dtype=tf.string)))
        new_ds = ds.map(lambda x, y: x)
        assert_val(first_element_of(new_ds), 1)

        # func: of tf ops
        add_func = lambda x,y: y
        new_ds = ds.map(add_func)
        assert_equal_tf(first_element_of(new_ds), 'foo')


    def test_filter(self):
        ds = tf.data.TextLineDataset('data/example1.txt')

        rm_comment_filter = lambda line: tf.not_equal(tf.strings.substr(line, 0, 1), '#')
        new_ds = ds.filter(rm_comment_filter)

        print(list(ds.as_numpy_iterator()))
        print()
        print(list(new_ds.as_numpy_iterator()))

    def test_batch(self):
        '''
            batch: stack n consecutive elements into one element
                before: element is e
                after:  element is [e, .., e]
        '''
        ds =tf.data.Dataset.range(100)
        batch_ds = ds.batch(2)

        assert_equal(first_element_of(ds), 0)
        assert_equal(first_element_of(batch_ds), [0,1])
