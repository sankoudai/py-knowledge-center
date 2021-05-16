__author__ = 'quiet road'
import unittest
import time
import math
from printutil.printutil import printVar

class WithTest(unittest.TestCase):
    """
        with contextmanager as var:
            used with contextmanager

        contextmanager class:
            A class which defines __enter__ and __exit__
            __enter__ is called before entering with-block
            The object returned by __enter__ will be assigned to var

            __exit__ is called at end of with-block
            If exception is raised from with-block, __exit__ is called before it propagages.
            And if __exit__ returned true, exception would be surpressed

        builtin contextmanager：
            file object
    """

    def setUp(self):
        class Timer(object):
            def __init__(self):
                self.interval = None

            def __enter__(self):
                self.start = time.clock()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.end = time.clock()
                self.interval = self.end - self.start

            def get_interval(self):
                return self.interval

        self.contextmanager = Timer()

    def test_with(self):
        printVar(self.contextmanager.get_interval())
        with self.contextmanager as timer:
            for i in range(0, 10000):
                if i % 1000 == 0:
                    print(i)
        printVar(timer.get_interval())


if __name__ == '__main__':
    unittest.main()
