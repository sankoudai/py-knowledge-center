__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class NumberTest(unittest.TestCase):
    def setUp(self):
        self.int_value = 1
        self.float_value = 1.0

    def type_test(self):
        print(isinstance(self.int_value, int))
        print(isinstance(self.float_value, float))

    def value_test(self):
        printVar(self.int_value)
        printVar(self.float_value)

    def int_operation_test(self):
        # + - * // results in int
        printVar(1 + 2)
        printVar( 3 // 2)  # 1
        # / results in float
        printVar(4 / 2)
        printVar(2 / 3)
        printVar(3 / 2)

    def float_operation_test(self):
        #always result in float
        printVar(1.0 / 0.2)

    def mix_operation_test(self):
        #mixed operation always result in float
        printVar(1 + 2.0)

    def double_slash_operator_test(self):
        """ test // operator: always round leftward"""
        printVar(3//2)  # 1
        printVar((-3) // 2)  # -2
        printVar(3 // (-2))  # -2

        printVar( 3.0 // 2)  # 1.0
        printVar((-3.0) // 2)  # -2.0

        printVar( 3 // 2.0)  # 1.0

    def mod_test(self):
        printVar( 3 % 2)  # int  1
        printVar( (-3) % 2)  # int  1
        printVar( 3 % (-2))  # int  -1
        printVar( (-3) % (-2))  # int -1
        print("------------------------")

        printVar(3.2 % 2)  # float 1.2
        printVar((-3.2) % 2) # float 0.8
        printVar(3.2 % (-2)) # -0.8
        printVar((-3.2) % (-2)) # 1.2

    def power_test(self):
        printVar(2 ** 3)  # int - 8
        printVar(1.1 ** 2)  # float - 1.2100..
        printVar(1.1 ** 2.0)  # float - 1.2100..

    def coercing_test(self):
        printVar(float(2))  # 2.0
        printVar(int(2.5))  # 2
        printVar(int(-2.5)) # -2.0


if __name__ == '__main___' :
    unittest.main()
