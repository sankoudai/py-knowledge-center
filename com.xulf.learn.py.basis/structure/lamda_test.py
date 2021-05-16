__author__ = 'quiet road'
import unittest
from printutil.printutil import printVar

class LamdaTest(unittest.TestCase):
    def setUp(self):
        self.incre_lambda = lambda x: x + 1
        self.sum_lambda = lambda x, y : x + y

    def test_type(self):
        '''lambda is short-hand function'''
        printVar(self.incre_lambda)
        printVar(self.sum_lambda)

    def test_use(self):
        printVar(self.incre_lambda(2))
        printVar(self.sum_lambda(2, 3))

if __name__ == '__main__':
    unittest.main()