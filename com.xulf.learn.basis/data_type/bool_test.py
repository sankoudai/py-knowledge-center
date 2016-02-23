__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class BoolTest(unittest.TestCase):
    def setUp(self):
        self.true_value = True
        self.false_value = False

    def type_test(self):
        print(isinstance(self.true_value, bool))

    def value_test(self):
        printVar(self.true_value)
        printVar(self.false_value)

    def relation_expr_test(self):
        printVar(1>0)
        printVar(1==0)

    def logical_expr_test(self):
        printVar(self.false_value and self.true_value)
        printVar(self.false_value or self.false_value)
        printVar(not self.false_value)

if __name__ == '__main___' :
    unittest.main()
