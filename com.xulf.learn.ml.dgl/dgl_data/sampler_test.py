import dgl
import torch
from unittest import TestCase
from dgl.dataloading import Sampler, BlockSampler

class SamplerTest(TestCase):
    '''
       Sampler: generate subgraph samples from original graph.
       Dataloader: iterables over subgraph samples.
       Together they form the basic primitives of the data pipeline
    '''

    def test_sampler(self):
        '''
            dgl.dataloading.Sampler：Sampler抽象基类， 核心方法sample(g, indices)
        '''
        class NodeSubgSampler(Sampler):
            def __init__(self):
                super(NodeSubgSampler, self).__init__()
            def sample(self, g, nids):
                return dgl.node_subgraph(g, nids)

        class EdgeSubgSampler(Sampler):
            def __init__(self):
                super(EdgeSubgSampler, self).__init__()
            def sample(self, g, eids):
                return dgl.edge_subgraph(g, eids)

        # homegeneous graph: 0-1-2-3-4-5
        g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]))

        sampler = NodeSubgSampler()
        subg = sampler.sample(g, [0,1, 4])
        assert subg.num_nodes() == 3
        assert subg.num_edges() == 1

        sampler = EdgeSubgSampler()
        subg = sampler.sample(g, eids=[0, 1])
        assert subg.num_nodes() == 3
        assert subg.num_edges() == 2

    def test_blocksampler(self):
        '''
        基类: BlockSampler(prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None)
            从原图采样mfgs的Sampler抽象基类
            构造参数:
                - prefetch_node_feats: fname_list or {ntype: fname_list} to fill 1st mfg's srcdata
                -prefetch_labels: fname_list or {ntype: fname_list} to fill last mfg's dstdata
                -prefetch_edge_feats:fname_list or {etype: fname_list} to fill all mfgs' edata

            方法:
            sample_blocks(g, seed_nodes, exclude_eids=None): sample mfgs的抽象方法，该方法实现采样即可（特征会自动由sample方法加上)
                - return: input_nodes, output_nodes, blocks
                    input_nodes: nids or {ntype:nids}, 第一层gnn module的mini-batch输入节点
                    output_nodes: nids, 最后一层gnn的输出节点
                    blocks: mfgs (理解：逻辑上说input_nodes应该是blocks[0]的src_nodes, out_nodes应该是blocks[-1]的dst_nodes

            sample(g, seed_nodes, exclude_eids=None): 调用sample_blocks，并将给mfgs附上指定特征

        子类一NeighborSampler
            NeighborSampler(fanouts, prob=None, prefetch_node_feats=None, ..)
            多层gnn的邻居采样器
            参数:
                - fanouts: [n] or [{etype:n}], ith-element being fanout for ith gnn layer
                - prob: str,如指定， 依g.edata[prob]概率采样
                - 其他字段BlockSampler通用字段
            Note: 与dgl.sampling.sample_neighbors相似

        子类二: MultiLayerFullNeighborSampler
            NeighborSampler的子类，获取所有的邻居节点，等价于NeighborSampler([-1]*num_layers, ...)
            参数差异：fantouts --> num_layers
        '''

        # tree: 0-1,2,  1-3,4, 2-5,6
        g = dgl.graph(([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2]))
        g.edata['prob'] = torch.FloatTensor([0., 1., 0., 1., 1.,0.])

        #NeighborSampler
        sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1], prob='prob')
        input_nodes, output_nodes, blocks = sampler.sample(g, seed_nodes=torch.tensor([0]))
        assert blocks[0].num_src_nodes() == 7
        assert blocks[0].num_dst_nodes() == 3
        assert blocks[1].num_src_nodes() == 3
        assert blocks[1].num_dst_nodes() == 1
        assert input_nodes.shape[0] == 7
        assert output_nodes.shape[0] == 1

        #MultiLayerFullNeighborSampler
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        input_nodes2, output_nodes2, blocks2 = sampler.sample(g, seed_nodes=torch.tensor([0]))
        torch.equal(input_nodes2, input_nodes)
        torch.equal(output_nodes2, output_nodes)

        assert blocks2[0].num_src_nodes() == blocks[0].num_src_nodes()
        assert blocks2[0].num_dst_nodes() == blocks[0].num_dst_nodes()
        assert blocks2[1].num_src_nodes() == blocks[1].num_src_nodes()
        assert blocks2[1].num_dst_nodes() == blocks[1].num_dst_nodes()