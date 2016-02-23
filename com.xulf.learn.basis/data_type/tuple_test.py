__author__ = 'quiet road'
import unittest


class TupleTest(unittest.TestCase):
    ''' tuple is immutable list '''

    def setUp(self):
        self.string_tuple = ('summer', 'is', 'at', 'its', 'uttermost')
        self.int_tuple = (2, 3, 7, 12)
        self.mixed_tuple = ('spring', 1, 'summer', 2, 'autumn', 3, 'winter', 4)
        self.single_ele_tuple = (1,)

    def type_test(self):
        print(isinstance(self.string_tuple, tuple))

    def test(self):
        print(self.string_tuple[0])
        print(self.string_tuple[0:2])
        print(self.string_tuple.index('is'))
        print(self.string_tuple.count('is'))

    def bool_context_test(self):
        # empty list is false
        if(not ()):
            print("Empty tuple is False in boolean context")

        print("All other tuple is True in boolean context")


if __name__ == '__main___' :
    unittest.main()
