__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class ComprehensionTest(unittest.TestCase):
    def setUp(self):
        self.seq = [1, 2, 2, 3]
        self.dict = {"one":1, "two":2}

    def list_comprehension_test(self):
        local_list = [2*item for item in self.seq]
        printVar(local_list)

    def set_comprehension_test(self):
        local_set = {item for item in self.seq}
        printVar(local_set)

    def dict_comprehension_test(self):
        reversed_dict = {val:key for key, val in self.dict.items()}
        printVar(reversed_dict)

    def generator_test(self):
        local_generator = (2*item for item in self.seq)
        printVar(local_generator)

    def if_comprehension_test(self):
        filtered_list = [i for i in self.seq if i != 2]
        printVar(filtered_list)


if __name__ == '__main__':
    unittest.main()