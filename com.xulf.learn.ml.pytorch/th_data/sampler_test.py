from unittest import TestCase
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler
from examples import ExampleTensorDataset

class SamplerTest(TestCase):
    '''
    Sampler: iterable over indices of dataset elements
        constructor argument: 基本接受一个dataset
        base methods: __iter__, __len__

    Note: BatchSampler is a sampler which returns a mini-batch of indices
    '''
    def test_sampler(self):
        # TensorDataset: 每个元素是个tuple
        t0 = torch.tensor([2, 1, 2, 3, 4])
        t1 = torch.tensor([0, 1, 0, 1, 0])
        ds = TensorDataset(t0, t1)
        assert next(iter(ds)) == (torch.tensor(2), torch.tensor(0))

        # SequentialSampler
        seq_sampler = SequentialSampler(ds)
        indxs = torch.tensor([i for i in seq_sampler])

        assert torch.equal(indxs, torch.tensor([0, 1, 2, 3, 4]))
        assert torch.equal(ds[indxs][0], t0)
        assert torch.equal(ds[indxs][1], t1)

        # RandomSampler
        sampler = torch.utils.data.RandomSampler(ds)
        indxs = torch.tensor([i for i in sampler])
        print('random sampler indxs: ', indxs)

        # BatchSampler(iterable, batch_size, drop_last)
        batch_sampler = torch.utils.data.BatchSampler(seq_sampler, 2, drop_last=True)
        batch_indxs = torch.tensor([batch for batch in batch_sampler])
        assert torch.equal(batch_indxs, torch.tensor([[0, 1], [2, 3]]))
        assert torch.equal(ds[batch_indxs][0], t0[batch_indxs])
        assert torch.equal(ds[batch_indxs][1], t1[batch_indxs])

