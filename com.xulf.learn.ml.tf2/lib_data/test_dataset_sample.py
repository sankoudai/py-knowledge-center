import tensorflow as tf
import numpy as np
from dataset_util import first_element_of
from cmp_util import *
import unittest
import math

np.set_printoptions(precision=3)


class TestDatasetSample(unittest.TestCase):
    '''
        test for ops that change dataset counts only
    '''

    def test_take(self):
        '''
            ds.take(count, name): creates a dataset with at most count elements
                count: if -1 or greater than ds size, contains all
        '''

        ds = tf.data.Dataset.range(4)
        assert_equal(first_element_of(ds), 0)

        sub_ds = ds.take(3)
        assert_equal_iterable(sub_ds, [0,1, 2])

        sub_ds = ds.take(10)
        assert_equal_iterable(sub_ds, [0, 1, 2, 3])

    def test_filter(self):
        ds = tf.data.Dataset.from_tensor_slices(['hello', 'world', '#comment'])

        rm_comment_filter = lambda line: tf.not_equal(tf.strings.substr(line, 0, 1), '#')
        new_ds = ds.filter(rm_comment_filter)
        assert_equal_iterable(new_ds, ['hello', 'world'])

    def test_rejection_resample(self):
        '''
            rejection_resample(class_func, target_dist, initial_dist=None, seed=None): resample to target distribution
                Resampling is performed via rejection sampling; some fraction of the input values will be dropped.
        '''
        # resample with reject
        ds = tf.data.Dataset.from_tensor_slices(([1.1, 1.2, 1.3, 1.4, 1.4, 1.4, 1.5],
                                                 [1, 0, 1, 1, 1, 1, 0]))
        resampler = tf.data.experimental.rejection_resample(
            lambda feat, label: label,
            [0.5, 0.5],
            initial_dist=[2 / 7, 5 / 7],
            seed=202112
        )

        sampled_ds = ds.apply(resampler)\
            .map(lambda extra_label, feat_label: feat_label)
        print(list(sampled_ds.as_numpy_iterator()))
