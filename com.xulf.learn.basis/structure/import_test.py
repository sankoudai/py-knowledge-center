__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class ImportTest(unittest.TestCase):
    def setUp(self):
        print("You can import sth")

    def test_direct_import(self):
        import math
        res = math.sqrt(4)
        printVar(res)

    def test_import_as(self):
        '''Use this as few as possible. '''
        import math as m
        printVar(m.sqrt(4))

    def test_import_elem(self):
        from math import sqrt
        printVar(sqrt(4))









if __name__ == '__main__':
    unittest.main()