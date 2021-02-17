import pandas as pd
import unittest

from utils.assert_util import assert_if


class AggTest(unittest.TestCase):
    def setUp(self) -> None:

        self.df = pd.DataFrame({
            'gender': ['f', 'f', 'f', 'f', 'm', 'm', 'm'],
            'age':    [11,  11,  10,  10,  10,  12,  10],
            'weight': [53,  51,  51,  50,  64,  63,  61]
        })

    def test_groupby(self):
        # one key
        grps = self.df.groupby("gender")
        for grp_key, grp in grps:
            v = grp['weight'].max()
            assert_if(grp_key == 'm', v == 64)
            assert_if(grp_key == 'f', v == 53)

        # multiple keys
        grps = self.df.groupby(['gender', 'age'])
        for grp_key, grp in grps:
            v = grp['weight'].max()
            assert_if(grp_key == ('f', 11), v == 53)
            assert_if(grp_key == ('f', 10), v == 51)