__author__ = 'quiet road'
import unittest


class AssertTest(unittest.TestCase):
    def setUp(self):
        assert 0 != 1

    def false_assert_test(self):
        '''False assert will crash the program'''
        assert 0 == 1
        print("You shouldn't be here")

    def true_test(self):
        assert 1== 1
        print("You must be here")



if __name__ == '__main__':
    unittest.main()