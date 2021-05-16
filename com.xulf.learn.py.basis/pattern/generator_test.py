__author__ = 'quiet road'
import unittest
import inspect

from printutil.printutil import printVar


class GeneratorTest(unittest.TestCase):
    """
    Generator Function:
        a function that uses yield
        returns a generator iterator.
        occupies 'constant memory'.
    Generator Iterator:
        an iterable （object with __next__(), __init__(), __iter__()）

    Execution:
        1.At first call, Run generator code until yield is executed .
            Yielded value is returned.
        2.At next call, generator code after yield is executed, util another yield is encountered.
                    Yielded value is returned.
        N. Again entering for loop, generator code ends without yield.
        Finally: StopIteration is thrown out. Iteration ends
    Extra:
        inspect.isgeneratorfunction(item)
            Return True if item is a generator, False otherwise

    """

    def setUp(self):
        def finite_generator_function():
            yield 1
            yield 2

        def infinite_generator_function():
            prev, cur = 0, 1
            while True:
                yield cur
                prev, cur = cur, prev + cur

        int_list = [1, 2, 3]
        self.finite_generator_function = finite_generator_function
        self.infinite_generator_function = infinite_generator_function
        self.comprehension_generator = (2 * item for item in int_list)

    def test_finite_generator(self):
        finite_generator = self.finite_generator_function()
        for item in finite_generator:
            printVar(item)

    def test_infinite_generator(self):
        infinite_generator = self.infinite_generator_function()
        for item in infinite_generator:
            if item > 10:
                break
            printVar(item)

    def test_comprehension_generator(self):
        for item in self.comprehension_generator:
            printVar(item)

    def test_isgeneratorfunction(self):
        printVar(inspect.isgeneratorfunction(self.finite_generator_function))


if __name__ == '__main__':
    unittest.main()
