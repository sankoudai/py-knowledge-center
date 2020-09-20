__author__ = 'quiet road'
import unittest

from printutil.printutil import printVar


class IteratorTest(unittest.TestCase):
    """
        Iteration:
            Process of taking items from sth, one after another.

        Iterable:
            Instance of class with a __iter__() method, which returns an iterator.

        Iterator:
            Instance of class with a __next__() method (next for python2),
            which returns the next value, raising StopIteration if no more.
    """
    def setUp(self):
        class Range(object):
            """Range simulates behavior of range
                It is an iterable-define class because it defines __iter__.
                It is an iterator-define class because it defines __next__
            """
            def __init__(self, end):
                assert end > 0
                self.cur = -1
                self.end = end

            def __iter__(self):
                return self

            def __next__(self):
                self.cur += 1
                if (self.cur < self.end):
                    return self.cur
                else:
                    raise StopIteration
        self.iterable = Range(10)
        self.iterator = iter(self.iterable)

    def test_iterable(self):
        """Under the hood, interpretor calls iter(self.iterable) for you
           This is equivalent to iterator_test.
        """
        for i in self.iterable:
            printVar(i)

    def test_iterator(self):
        """Under the hood, interpretor calls next(self.iterator) for you,
           which calls self.iterator.__next__() to get the next item
        """
        for i in self.iterator:
            printVar(i)

    def test_next(self):
        printVar(next(self.iterator))
        printVar(next(self.iterator))
        printVar(next(self.iterator))
        printVar(next(self.iterator))

if __name__ == '__main__':
    unittest.main()
