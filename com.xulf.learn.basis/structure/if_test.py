__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class ImportTest(unittest.TestCase):
    def setUp(self):
        '''The following values evaluates to Fase in boolean context.
            All others evaluates to True.'''
        self.bool_false = False
        self.none_false = None
        self.int_false = 0
        self.float_false = 0.0
        self.string_false = ''
        self.list_false = []
        self.tuple_false = ()
        self.set_false = set()
        self.dict_false = {}

    def test_false(self):
        if not self.bool_false:
            print('bool : False is false in boolean context')
        if not self.none_false:
            print('NoneType: None is false in boolean context')
        if not self.int_false:
            print('int: 0 is false in boolean context')
        if not self.float_false:
            print('float: 0.0 is false in boolean context')
        if not self.string_false:
            print('string: "" is false in boolean context')
        if not self.list_false:
            print('list: [] is false in boolean context')
        if not self.tuple_false:
            print('tuple: () is false in boolean context')
        if not self.set_false:
            print('set: set() is false in boolean context')
        if not self.dict_false:
            print('dict: {} is false in boolean context')
    
    def test_equal(self):
        '''== means same appearance
            1.x == y calls x.__eq__(y)
              x != y calls x.__ne__(y)
            2.Always use == on basic values: bool, number, string
              Always use == on immutable valus: string
        '''
        x = 'abc'
        y = 'abc'
        if x == y:
            print('== means same appearance')

    def test_order(self):
        '''1. x < y calls x.__lt__(y)
              x <= y calls x.__le__(y)
              x > y calls x.__gt__(y)
              x >= y calls x__ge__(y)
          2. string is ordered lexicographically
        '''
        x = 'abc'
        y = 'bc'
        if(y > x):
            print('string is ordered lexicographically')

    def test_identity(self):
        '''is means same identity
           is can't be customized
        '''
        x = [1, 2 ,3]
        y = [1, 2, 3]

        print("x is y: {}".format(x is y)) # False
        print("x == y: {}".format(x == y)) # True

    def test_else(self):
        if False:
            print("You won't be here")
        else:
            print("You should be here")

    def test_elif(self):
        num = 4
        if num < 0:
            print("No")
        elif num < 2:
            print("No")
        else:
            printVar(num)

    def test_short_circuit_logic(self):
        '''You can use short circuit in expressions other than bool'''
        res = 'jim' or "Unknown"
        printVar(res)

if __name__ == '__main__':
    unittest.main()