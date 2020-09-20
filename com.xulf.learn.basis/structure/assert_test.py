__author__ = 'quiet road'
import unittest


class AssertTest(unittest.TestCase):
    def setUp(self):
        assert 0 != 1

    def test_false_assert(self):
        '''False assert will crash the program'''
        assert 0 == 1
        print("You shouldn't be here")

    def test_true(self):
        assert 1== 1
        print("You must be here")



if __name__ == '__main__':
    unittest.main()