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

           内存数据（如numpy array)一般使用from_tensor_slices转化为
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
        print(dataset.element_spec)

    def test_from_generator(self):
        '''
            tf.data.Dataset.from_generator(generator, args, output_signature):
                - generator
                - args:  a tuple, generator(args)
                - output_signature: A (nested) structure of tf.TypeSpec objects

            使用场景：比如当内存装不下
        '''
        # dataset from generator
        def gen_train_data(cnt):
            for i in range(cnt):
                yield tf.random.uniform([3], seed=100), np.random.randint(0, 2, size=1)
        ds = tf.data.Dataset.from_generator(gen_train_data, args=[100],
                                            output_signature=(tf.TensorSpec(shape=(3,), dtype=tf.float32),
                                                              tf.TensorSpec(shape=(1,), dtype=tf.int32)))
        ds = ds.shuffle(100).batch(10)

        # usage example
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.optimizers.SGD())
        model.fit(ds, epochs=10)

    def test_TextLineDataset(self):
        # read one file
        ds = tf.data.TextLineDataset('data/example_text.txt')
        print(first_element_of(ds))

        # read multiple files
        ds = tf.data.TextLineDataset(['data/example_text2.txt', 'data/example_text.txt'])
        print(first_element_of(ds))

    def test_load_csv(self):
        # high level: make_csv_dataset
        ds = tf.data.experimental.make_csv_dataset('../data/titanic/train.csv',
                                                   batch_size=2,
                                                   label_name='survived',
                                                   select_columns=['class', 'fare', 'survived'],
                                                   shuffle_seed=100)
        feature_batch, label_batch = first_element_of(ds)
        assert_equal(feature_batch['fare'], [77.9583, 19.5], tol=0.1)
        assert_equal(label_batch, [1, 1])


    def test_zip(self):
        ## 综合示例：param是(dict, tuple)
        features = tf.data.Dataset.from_tensor_slices({'f1': np.array([1, 2, 3]), 'f2':np.array([4, 5, 6])})
        labels = tf.data.Dataset.from_tensor_slices(np.array([1, 0, 1]))

        dataset = tf.data.Dataset.zip((features, labels))
        elem = first_element_of(dataset)
        assert_equal_dict(elem[0], {'f1':1, 'f2':4})
        assert_val(elem[1], 1)


