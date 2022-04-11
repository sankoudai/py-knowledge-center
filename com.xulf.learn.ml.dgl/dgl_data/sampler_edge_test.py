import dgl
import torch
from unittest import TestCase
from dgl.dataloading import EdgePredictionSampler

class EdgeSamplerTest(TestCase):
    def test_edge_pred_sampler(self):
        '''
            EdgePredictionSampler: Sampler的子类， 通过seed_eids采样 input_nodes, pos_graph, neg_graph, blocks
            EdgePredictionSampler(sampler, negative_sampler=None, exclude=None, reverse_eids=None, prefetch_labels=None ..)
            参数说明：
                - sampler： 一个node sampler
                - negative_sampler: negative_sampler，for sample the neg_graph
                - exclude: str,
                    self -- remove edges in minibatch
                    reverse_id -- remove minibatch edges and their reverse edge stored in reverse_eids argument
                - reverse_eids: {tensor_eids} or {etype:tensor_eids}, ith elem being ith-edge's reverse edge
                - prefetch_labels: fname_list or {etype:fname_list} to the sampled posive graph
            核心方法：
                sample(g, seed_edges)
                参数：- seed_edges: tensor_eids or {etype:tensor_eids},  also called minibatch edges
                返回值： input_nodes, pos_graph, neg_graph, blocks
                    - pos_graph:  edge_subgraph induced by seed_edges (会与neg_graph一起compact)
                    - neg_graph: graph with faked edges: negative_sampler(g, seed_edges)
                    - input_nodes, _, blocks =  sampler(g, pos_graph.ndata[NID], exclude_eids)
                    (Note: neg_graph and pos_graph are compacted so that they have the same nodes)
        '''
        # tree: 0-1,2,  1-3,4, 2-5,6
        g = dgl.graph((torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([0, 0, 1, 1, 2, 2])))
        seed_eids = torch.tensor([0, 3])
        seed_nids = torch.tensor([0, 1, 3]) # incident nodes of seed_eids

        # edge sampler
        neg_sampler = dgl.dataloading.negative_sampler.Uniform(3)
        node_sampler = dgl.dataloading.NeighborSampler(fanouts=[-1, -1], prob='prob')
        edgepred_sampler = dgl.dataloading.EdgePredictionSampler(
            node_sampler,
            negative_sampler=neg_sampler,
            exclude='self')

        input_nodes, pos_graph, neg_graph, blocks = edgepred_sampler.sample(g, seed_eids)
        assert pos_graph.num_edges() == 2
        assert pos_graph.num_nodes() == neg_graph.num_nodes()


