import unittest
import tensorflow as tf
import numpy as np
from dataset_util import first_element_of
from cmp_util import *
import math

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
        power_func_tf = lambda x, y: (tf.pow(2, x), y)
        new_ds = ds.map(power_func_tf)
        assert_equal_iterable(first_element_of(new_ds), (2, 'foo'))

        # func: of python op
        power_func_py = lambda x, y: (tf.py_function(math.pow, (2, x), tf.float32), y)
        new_ds = ds.map((power_func_py))
        assert_equal_iterable(first_element_of(new_ds), (2., 'foo'))

    def test_filter(self):
        ds = tf.data.TextLineDataset('data/example1.txt')

        rm_comment_filter = lambda line: tf.not_equal(tf.strings.substr(line, 0, 1), '#')
        new_ds = ds.filter(rm_comment_filter)

        print(list(ds.as_numpy_iterator()))
        print()
        print(list(new_ds.as_numpy_iterator()))

    def test_apply(self):
        """
            ds.apply(trans_func):
            - trans_func: a function with ds argument and ds return
        """
        ds = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])

        def trans_func(ds):
            return ds.filter(lambda x: x > 2)

        new_ds = ds.apply(trans_func)
        assert_equal(first_element_of(new_ds), 3)

    def test_batch(self):
        """
            batch: stack n consecutive elements into one element
                before: element is e
                after:  element is [e, .., e]
        """
        ds = tf.data.Dataset.range(100)
        assert_equal(first_element_of(ds), 0)

        # batch
        batch_ds = ds.batch(2)
        assert_equal(first_element_of(batch_ds), [0, 1])

        # unbatch
        new_ds = batch_ds.unbatch()
        assert_equal(first_element_of(new_ds), 0)

    def test_resample(self):
        # resample with reject
        ds = tf.data.Dataset.from_tensor_slices(([1.1, 1.2, 1.3, 1.4, 1.4, 1.4, 1.5],
                                                 [1, 0, 1, 1, 1, 1, 0]))
        resampler = tf.data.experimental.rejection_resample(
            lambda feat, label: label,
            [0.5, 0.5],
            initial_dist=[2 / 7, 5 / 7]
        )

        batch_ds = ds.apply(resampler) \
            .map(lambda extra_label, feat_label: feat_label) \
            .batch(10)
        print(first_element_of(batch_ds))
