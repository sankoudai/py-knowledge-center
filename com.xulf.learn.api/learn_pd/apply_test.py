import unittest
import pandas as pd

class ApplyTest(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

    def test_apply(self):
        pass
