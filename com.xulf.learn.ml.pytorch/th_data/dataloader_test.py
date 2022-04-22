import torch
from torch.utils.data import DataLoader
from unittest import TestCase
from examples import ExampleTensorDataset

class DataLoaderTest(TestCase):
    '''
        DataLoader： 概念模型是iterator，搭配Dataset使用（iterable/map)
        提供的功能：
            - 迭代器功能 （base）
            - 转换功能 (可选) -- collate_fn
            - 加载顺序 (可选）
            - batch处理功能 （可选)
            - 加载性能优化 （可选）

        iterable-stype dataset的有效参数： DataLoader(iter_ds, collate_fn=None, batch_size=1, shuffle=False,..)
        - 加载顺序： 通过shuffle指定
        - batch功能：通过batch_size, drop_last实现
        - collate_fn: 基本等价于
            ```
            dataset_iter = iter(dataset)
            for indices in batch_sampler:
                yield collate_fn([next(dataset_iter) for _ in indices])
            ```

        map-style dataset的有效参数：DataLoader(kv_ds, collate_fn=None, sampler=None, batch_size=1, drop_last=False, batch_sampler=None,..)
        - 加载顺序/batch： sampler+batch_size+drop_last 或者 batch_sampler
        - collate_fn: 基本等价于
           ```
                for indices in batch_sampler:
                    yield collate_fn([dataset[i] for i in indices])
            ```
        说明：
            1）sampler 一般返回的是单个样本(或其keys)， 一般与batch_size, drop_last 配合使用
            2) batch_sampler：  一般返回样本batch(或其keys)； （与sampler, batch_size, drop_last 互斥)

    '''
    def test_dataloader(self):
        # TensorDataset: 每个元素是个tuple, map-style dataset
        t = torch.tensor([2, 1, 2, 3, 4])
        ds = ExampleTensorDataset(t)
        assert next(iter(ds)) == torch.tensor(2)

        # mode1 : sampler+batch_size+drop_last
        # 1) sampler: 产出indxs = [0, 1, 2, 3, 4]
        # 2) batch_indxs = [[0,1], [2,3]]
        # 3) 对每个batch_indx： tensor_list = [ds[i] for i in batched_idx] -- 注这里是单条数据的list !
        # 4) 转换: data = collate_fn(tensor_list) (collate_fn=None时， 默认会将list stack成一个tensor)
        # 5) 返回data
        seq_sampler = torch.utils.data.SequentialSampler(ds) # shuffle=False时，DataLoader内部也是使用这个sampler
        collate_fn = lambda x : torch.stack(x, 0) # collate_fn=None时，DataLoader内部使用这个转化
        dataloader = DataLoader(ds,
                                collate_fn=collate_fn,
                                sampler=seq_sampler, batch_size=2, drop_last=True)
        batches = [batch for batch in dataloader]
        assert torch.equal(batches[0], torch.tensor([2, 1]))
        assert torch.equal(batches[1], torch.tensor([2, 3]))

        # mode2 : batch_sampler=batch_sampler
        # collate_fn: 输入是tensor_list（[ds[i] for i in batch_indx])
        batch_sampler = torch.utils.data.BatchSampler(seq_sampler, batch_size=2, drop_last=True)
        dataloader = DataLoader(ds, collate_fn=collate_fn, batch_sampler=batch_sampler)
        batches = [batch for batch in dataloader]

        assert torch.equal(batches[0], torch.tensor([2, 1]))
        assert torch.equal(batches[1], torch.tensor([2, 3]))

        # mode3: sampler=batch_sampler, batch_size=None
        # collate_fn: 输入是tensor（ds[batch_indx])
        collate_fn_tensor = lambda x : x+1
        dataloader = DataLoader(ds,
                                collate_fn=collate_fn_tensor,
                                sampler=batch_sampler, batch_size=None)
        batches = [batch for batch in dataloader]

        assert torch.equal(batches[0], torch.tensor([2, 1]))
        assert torch.equal(batches[1], torch.tensor([2, 3]))