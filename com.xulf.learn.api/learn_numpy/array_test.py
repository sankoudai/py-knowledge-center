__author__ = 'admin'
import unittest
import numpy as np

from printutil.printutil import printVar


class ArrayTest(unittest.TestCase):
    def setUp(self):
        # self.num_arr = np.array
        pass

    def test_create_1d(self):
        # from sequence
        arr = np.array([0, 3, 4])
        printVar(arr)

        arr = np.array((1, 3, 4))
        printVar(arr)

    def test_create_nd(self):
        # 2 X 3
        arr = np.array([[0, 1, 2], [3, 4, 5]])
        self.print_attr(arr)
        print()

        # 2 X 2 X 2
        arr = np.array([[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]])
        self.print_attr(arr)
        printVar(arr)

    def test_create_byfunc(self):
        # even spaced 1d
        arr = np.arange(10)
        printVar(arr)

        arr = np.arange(0, 11, 2)  # start, end(exclusive), step
        printVar(arr)

        arr = np.linspace(0, 11, 5)  # start, end, number of points
        printVar(arr)

        # ones && zeros
        arr = np.ones(3)
        printVar(arr)
        arr = np.ones((2, 2))
        printVar(arr)

        arr = np.zeros(3)
        printVar(arr)
        arr = np.zeros((2, 2))
        printVar(arr)

        # diagnal
        I = np.eye(3)
        printVar(I)

        diag = np.diag(np.array([1,2,3,4]))
        printVar(diag)

        #random
        arr = np.random.rand(4)       # uniform in [0, 1]
        printVar(arr)

        arr = np.random.randn(4)  # Gaussian
        printVar(arr)

    def test_dtype(self):
        # dtype: int32 int64 float64
        arr = np.array([1, 3], dtype='float64')
        print('dtype={}'.format(arr.dtype))
        printVar(arr)

    def test_attr(self):
        arr = np.array([0, 3, 5])
        self.print_attr(arr)
        print()

    def print_attr(self, arr):
        print('dim = {}'.format(arr.ndim))
        print('shape = {}'.format(arr.shape))
        print('len={}'.format(len(arr)))
        print('size={}'.format(arr.size))

