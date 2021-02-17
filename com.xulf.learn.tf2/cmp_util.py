import tensorflow as tf
import numpy as np

def assert_shape(tensor, shape):
    tensor_shape = tf.shape(tensor).numpy()
    assert len(tensor_shape) == len(shape)
    assert np.all(np.equal(tensor_shape, shape))

def assert_val(tensor, val):
    assert np.all(np.equal(tensor, val))

def assert_less(tensor, val):
    assert np.all(np.less(tensor, val))

def assert_greater(tensor, val):
    assert np.all(np.greater(tensor, val))

def assert_equal(tensor, val, tol=None):
    if tol is None:
        assert np.all(np.equal(tensor, val))
    else:
        assert np.all(np.less(np.abs(tensor-val), tol))

def assert_ne(tensor, val, tol=10e-6):
    assert np.all(np.greater(np.abs(tensor-val), tol))