__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class BoolTest(unittest.TestCase):
    def setUp(self):
        self.true_value = True
        self.false_value = False

    def test_type(self):
        print(isinstance(self.true_value, bool))

    def test_value(self):
        printVar(self.true_value)
        printVar(self.false_value)

    def test_relation_expr(self):
        printVar(1>0)
        printVar(1==0)

    def test_logical_expr(self):
        printVar(self.false_value and self.true_value)
        printVar(self.false_value or self.false_value)
        printVar(not self.false_value)

if __name__ == '__main___' :
    unittest.main()
