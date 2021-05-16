__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class ComprehensionTest(unittest.TestCase):
    def setUp(self):
        self.seq = [1, 2, 2, 3]
        self.dict = {"one":1, "two":2}

    def test_list_comprehension(self):
        local_list = [2*item for item in self.seq]
        printVar(local_list)

    def test_set_comprehension(self):
        local_set = {item for item in self.seq}
        printVar(local_set)

    def test_dict_comprehension(self):
        reversed_dict = {val:key for key, val in self.dict.items()}
        printVar(reversed_dict)

    def test_generator(self):
        local_generator = (2*item for item in self.seq)
        printVar(local_generator)

    def test_if_comprehension(self):
        filtered_list = [i for i in self.seq if i != 2]
        printVar(filtered_list)


if __name__ == '__main__':
    unittest.main()