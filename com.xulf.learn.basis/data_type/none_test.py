__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class NoneTest(unittest.TestCase):
    def setUp(self):
        self.a_none = None
        self.b_none = None

    def type_test(self):
        '''None is of NoneType, the only value'''
        print(type(self.a_none))

    def value_test(self):
        printVar(self.a_none)

    def equal_only_to_none_test(self):
        '''None equals only to none'''
        print(self.a_none == self.b_none)  # True
        print(self.a_none == '')  # False

    def bool_context_test(self):
        # None evalues to Fasle, not None evaluates to True
        if not None:
            print('None evalues to Fasle, not None evaluates to True')


if __name__ == '__main___' :
    unittest.main()
