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
    if type(val) == str:
        assert np.all(tf.equal(tensor, val))
        return

    if tol is None:
        assert np.all(np.equal(tensor, val))
    else:
        assert np.all(np.less(np.abs(tensor-val), tol))

def assert_equal_ta(ta1, ta2, tol=None):
    t1 = ta1.stack()
    t2 = ta2.stack()
    assert_equal(t1, t2, tol)

def assert_equal_sparse(sparse_tensor, dense_tensor):
    t = tf.sparse.to_dense(sparse_tensor)
    assert_equal(t, dense_tensor)

def assert_equal_dict(d1, d2):
    for key, val in d1.items():
        assert key in d2
        assert_equal(val, d2[key])
    for key, val in d2.items():
        assert key in d1
        assert_equal(val, d1[key])

def assert_equal_iterable(iter1, iter2):
    for i, j in zip(iter1, iter2):
        assert_equal(i, j)

def assert_same(v1, v2):
    assert v1 is v2

def assert_ne(tensor, val, tol=10e-6):
    assert np.all(np.greater(np.abs(tensor-val), tol))

def assert_type(param, data_type):
    assert type(param) is data_type