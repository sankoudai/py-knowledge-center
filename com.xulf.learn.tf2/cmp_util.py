import tensorflow as tf
import numpy as np

def assert_shape(tensor, shape):
    tensor_shape = tf.shape(tensor).numpy()
    assert len(tensor_shape) == len(shape)
    assert np.all(np.equal(tensor_shape, shape))