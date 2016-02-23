__author__ = 'quiet road'
import unittest
from printutil.printutil import printVar

class MapTest(unittest.TestCase):
    """
        map:
            map is a class, which implements iterable pattern and iterator pattern.
            constructor:  map(f, iterator1, iterator2, ...)
    """

    def setUp(self):
        self.a_list = [1, 2, 3, 4]
        self.b_list = [4, 3, 2, 1]
        self.small_list = [1, 2]
        self.unary_f = lambda x : x + 1
        self.n_nary_f = lambda x, y : x + y

    def unary_map_test(self):
        map_object = map(self.unary_f, self.a_list)
        printVar(map_object)
        print()

        for item in map_object:
            printVar(item)

    def nnary_map_test(self):
        map_object = map(self.n_nary_f, self.a_list, self.b_list)
        printVar(map_object)
        print()

        for item in map_object:
            printVar(item)

    def diff_lengths_test(self):
        # map_object has size of self.small_list
        map_object = map(self.n_nary_f, self.a_list, self.small_list)
        printVar(map_object)
        print()

        for item in map_object:
            printVar(item)

if __name__ == '__main__':
    unittest.main()
