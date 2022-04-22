import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from unittest import TestCase

class DatasetTest(TestCase):
    '''
        two types of Dataset: map-style, iterable-style
        map-style: 基类是Dataset，代表kv类型的数据集
        iterable-style: 基类是IterableDataset, 代表list类型的数据集
    '''
    def test_dataset(self):
        '''
            Dataset: map-style抽象基类; 提供方法ds[key], len(ds)
            子类必须实现
                def __getitem__(self, index)
                def __len__(self)
        '''
        class ExampleDataset(Dataset):
            def __init__(self, size):
                self.size = size
                self.data = torch.arange(size) + 100

            def __getitem__(self, index):
                if torch.is_tensor(index):
                    index = index.tolist()
                return self.data[index]

            def __len__(self):
                return self.size

        ds = ExampleDataset(10)
        vals = ds[torch.tensor([2, 3])]
        assert torch.equal(vals, torch.tensor([102, 103]))
        assert len(ds) == 10

    def test_iteralbel_dataset(self):
        '''
        IterableDataset: iterable-style抽象类， 提供iter方法
        子类必须实现：
            __iter__()

        Note:
        一般在Dataloader进行多进程加载时， ds会被复制n份。 为避免数据重复，可通过get_worker_info()获取当前加载进程信息，进而做不同分片。
        '''

        class WorkerSplitDataset(IterableDataset):
            def __init__(self, start, end):
                super(WorkerSplitDataset, self).__init__()
                self.start = start
                self.end = end

            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                if worker_info is None: #单进程加载
                    iter_start = self.start
                    iter_end = self.end
                else:
                    per_worker = (self.end - self.start) // worker_info.num_workers + 1
                    worker_id = worker_info.id
                    iter_start = self.start + worker_id * per_worker
                    iter_end = min(iter_start + per_worker, self.end)

                return iter(range(iter_start, iter_end))

        ds = WorkerSplitDataset(start=3, end=7)

        #单独使用
        ds_iter = iter(ds)
        assert next(ds_iter) == 3
        assert next(ds_iter) == 4

        # 使用DataLoader进行单进程加载
        dataloader = DataLoader(ds, num_workers=0)
        print(list(dataloader))

        #使用DataLoader进行多进程加载：合并加载出[3,4 ,5 ,6]
        dataloader = DataLoader(ds, num_workers=2)
        print(list(dataloader))