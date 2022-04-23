import dgl
import torch
import numpy as np
from dgl.dataloading import DataLoader,NodeDataLoader, EdgeDataLoader
from unittest import TestCase

class DataloaderTest(TestCase):
    '''
            dgl.dataloading.DataLoader: pytorch DataLoader的子类
            DataLoader(g, indices,block_sampler, batch_size=1, drop_last=False  ..)

            构造参数：
                - indices: tensor or {type:tensor}, 在graph_sampler中使用
                - block_sampler: dgl.dataloading.BlockSampler, subgraph sampler
                (note：
                    1) indices + batch_size + drop_last => 构造一个节点的dataset
                    2） g + block_sampler ==> 构造collate_fn
                    3）两种取值模式：
                        节点任务：(nids + sampler) => input_nodes, out_nodes, blocks
                        边任务： (eids + EdgePredictionSampler)=> input_nodes, pos_graph, neg_graph, blocks
                )
        '''
    def test_dataloader(self):
        g = dgl.rand_graph(300, 300)


        # 用于节点任务
        train_nids = torch.tensor([10, 20, 30, 40, 50, 60])
        sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])

        batch_size = 3
        dataloader = dgl.dataloading.DataLoader(
            g, train_nids, sampler,
            batch_size=batch_size, shuffle=False
        )
        input_nodes, output_nodes, blocks = next(iter(dataloader))
        assert len(output_nodes) == batch_size
        assert blocks[-1].num_dst_nodes() == batch_size
        assert len(blocks) == 3

        # 用于边任务
        batch_size, neg_k = 2, 3

        train_eids = torch.tensor([0, 1, 2, 3])
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(neg_k)
        node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1], prob='prob')
        edge_sampler = dgl.dataloading.EdgePredictionSampler(
            node_sampler,
            negative_sampler=neg_sampler,
            exclude='self')

        dataloader = dgl.dataloading.DataLoader(g, train_eids, edge_sampler, batch_size=batch_size)
        input_nodes, pos_g, neg_g, blocks = next(iter(dataloader))
        assert pos_g.num_edges() == batch_size
        assert neg_g.num_edges() == batch_size * neg_k
        assert len(blocks) == 2