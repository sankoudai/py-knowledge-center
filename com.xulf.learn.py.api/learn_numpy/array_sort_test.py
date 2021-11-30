import unittest
import numpy as np
from cmp_util import *

class ArraySortTest(unittest.TestCase):
    def test_argsort(self):
        '''
            np.argsort(x, axis=0): 沿axis比较，将下标按从小到大返回
        '''
        x = np.array([1,4,3,-1,6,9])
        args = x.argsort()
        assert_equal(args, np.array([3, 0, 2, 1, 4, 5]))
        assert_val(x[args[0]], -1)
