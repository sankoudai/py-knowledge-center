__author__ = 'quiet road'
import unittest


class ClosureTest(unittest.TestCase):
    """
        Closure:
            A data structure storing a function and its environment at creation time.
            The environment is mostly a map storing variables defined in enclosing scope.
    """
    def setUp(self):
        def incrementor_producer(step):
            def increment_by(x):
                return x + step
            return increment_by
        self.incre_closure = incrementor_producer(1) # incre_closure has with it the snapshot of step = 1
        self.incre_by_2_closure = incrementor_producer(2)

    def test(self):
        assert self.incre_closure(2) == 3
        assert self.incre_by_2_closure(2) == 4





if __name__ == '__main__':
    unittest.main()
