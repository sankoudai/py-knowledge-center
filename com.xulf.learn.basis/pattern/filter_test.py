__author__ = 'quiet road'
import unittest
from printutil.printutil import printVar

class FilterTest(unittest.TestCase):
    """
        filter:
            filter is a class, which implements iterable pattern and iterator pattern.
            constructor:  filter(bool_f, iterator)
    """

    def setUp(self):
        self.a_list = [1, 2, 3, 4]
        self.bool_f = lambda x : x % 2 ==0

    def use_test(self):
        filter_object = filter(self.bool_f, self.a_list)
        printVar(filter_object)
        print()

        for item in filter_object:
            printVar(item)

    def modify_iterator_test(self):
        '''Modifying underlying iterator affects the resulting filter_object,
            even after filter_object is created.
        '''
        filter_object = filter(self.bool_f, self.a_list)
        self.a_list[0] = 6

        for item in filter_object:
            printVar(item)

    def list_test(self):
        filter_object = filter(self.bool_f, self.a_list)
        local_list = list(filter_object)
        printVar(local_list)

if __name__ == '__main__':
    unittest.main()
