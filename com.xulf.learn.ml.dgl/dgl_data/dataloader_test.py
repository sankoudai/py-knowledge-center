import dgl
import torch
from dgl.dataloading import NodeDataLoader
from unittest import TestCase

class DataloaderTest(TestCase):
    def test_node_dataloader(self):
        '''
            NodeDataLoader(g, nids, block_sampler, ..)
            参数：
                - nids: tensor or {ntype:tensor}, 在graph_sampler中使用
                - block_sampler: dgl.dataloading.BlockSampler, subgraph sampler
        '''
        g = dgl.rand_graph(300, 3000)
        train_nids = torch.tensor([0, 1, 2, 3, 4, 5])
        sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
        dataloader = dgl.dataloading.NodeDataLoader(
            g, train_nids, sampler,
            batch_size=3, shuffle=True,  num_workers=1)
        for input_nodes, output_nodes, blocks in dataloader:
            print('-----')
            print(input_nodes)
            print(output_nodes)
            print(blocks)

    # def test_edge_dataloader(self):
