from unittest import TestCase
from collections import namedtuple
from collections import OrderedDict
from itertools import product
class TestNamedtuple(TestCase):
    '''
            Usage:
                1. define type:  T = namedtuple(name, fieldnames)
                2. define instance:  t = T(field_values)
                3. use instance: t.xx , where xx is a fieldname
        '''
    def test_construction(self):


        Run = namedtuple('Run', ('batch', 'lr'))
        run = Run(100, 0.1)
        assert run.lr==0.1
        assert run.batch==100

    def test_usecase(self):
        # package parameters
        def get_runs(params):
            Run = namedtuple('Run', params.keys())
            runs = []
            for v in product(*params.values()):
                runs.append(Run(*v))
            return runs

        params = OrderedDict(
            lr = [0.1, 0.01],
            batch = [10, 100, 1000]
        )
        for run in get_runs(params):
            print(run)