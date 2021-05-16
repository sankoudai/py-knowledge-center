__author__ = 'quiet road'
import unittest
from functools import reduce

from printutil.printutil import printVar

class ReduceTest(unittest.TestCase):
    """
        reduce:
            reduce(func_xy, sequence[, initial]) –> value
            where
                func_xy(x, y) is two-parameter function
    """
    def test_reduce(self):
        val = reduce(lambda x,y:x+y, [1, 2, 3])
        printVar(val)
