__author__ = 'quiet road'
import unittest


class ClassTest(unittest.TestCase):
    """
        class:
            special method:
               __init__(self, ...): called at instantiation
    """

    def init_test(self):
        instance = Class("jim")
        instance.print_name()
        instance.call_method()

class Class(object):
    def __init__(self, name):
        self.name = name

    def print_name(self):
        print("name: " + self.name)

    def call_method(self):
        self.print_name()

if __name__ == '__main__':
    unittest.main()
