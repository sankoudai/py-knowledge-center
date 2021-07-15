import tensorflow as tf
import unittest
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from dataset_util import first_element_of
from cmp_util import *

np.set_printoptions(precision=4)

class TestDatasetDef(unittest.TestCase):
    '''
        tf.data.Dataset: element序列的抽象, element可以是Tensor, tuple, dict, NamedTuple, Ordered
    '''

    def test_from_tensor_slices(self):
        '''
           tf.data.Dataset.from_tensor_slices(param):
           param有以下情形：
              a) tuple： element是同长tuple (of tensor)
              b) dict:  element是同key的dict (of tensor)
              c) list or np.array:  element是tensor
        '''
        #param是list or np.array
        dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        assert_equal(first_element_of(dataset), 1)

        dataset = tf.data.Dataset.from_tensor_slices([[1, 2, 3], [4, 5, 6]])
        assert_equal(first_element_of(dataset), [1, 2, 3])

        dataset = tf.data.Dataset.from_tensor_slices(np.array([[1, 2, 3], [4, 5, 6]]))
        assert_equal(first_element_of(dataset), [1, 2, 3])

        #param是tuple
        dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3], [4, 5, 6]))
        assert_equal(first_element_of(dataset), (1, 4))

        #param是dict
        dataset = tf.data.Dataset.from_tensor_slices({'f1': [1, 2, 3], 'f2':[4, 5, 6]})
        print(first_element_of(dataset))
        assert_equal_dict(first_element_of(dataset), {'f1':1, 'f2':4})

        ## 综合示例：param是(dict, tuple)
        features = {'f1': np.array([1, 2, 3]), 'f2':np.array([4, 5, 6])}
        labels = np.array([1, 0, 1])
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))

        elem = first_element_of(dataset)
        assert_equal_dict(elem[0], {'f1':1, 'f2':4})
        assert_val(elem[1], 1)

    def test_zip(self):
        ## 综合示例：param是(dict, tuple)
        features = tf.data.Dataset.from_tensor_slices({'f1': np.array([1, 2, 3]), 'f2':np.array([4, 5, 6])})
        labels = tf.data.Dataset.from_tensor_slices(np.array([1, 0, 1]))

        dataset = tf.data.Dataset.zip((features, labels))
        elem = first_element_of(dataset)
        assert_equal_dict(elem[0], {'f1':1, 'f2':4})
        assert_val(elem[1], 1)

