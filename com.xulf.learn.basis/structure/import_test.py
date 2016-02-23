__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class ImportTest(unittest.TestCase):
    def setUp(self):
        print("You can import sth")

    def direct_import_test(self):
        import math
        res = math.sqrt(4)
        printVar(res)

    def import_as_test(self):
        '''Use this as few as possible. '''
        import math as m
        printVar(m.sqrt(4))

    def import_elem_test(self):
        from math import sqrt
        printVar(sqrt(4))









if __name__ == '__main__':
    unittest.main()