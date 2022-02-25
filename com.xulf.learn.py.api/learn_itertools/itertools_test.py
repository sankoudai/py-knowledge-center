from unittest import TestCase
from itertools import product
from printutil.printutil import *

class ItertoolsTest(TestCase):
    def test_product(self):
        '''
            product: 返回一个iterator, 结果为笛卡尔积
        '''
        dim1 = ['a', 'b', 'c']
        dim2 = [True, False]
        dim3 = [1, 2]

        p = product(dim1, dim2, dim3)
        assert next(p) == ('a', True, 1)

        for d1, d2, d3 in product(dim1, dim2, dim3):
            print(d1, d2, d3)
