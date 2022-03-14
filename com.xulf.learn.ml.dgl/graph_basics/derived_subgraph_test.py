import dgl
import torch
from unittest import TestCase

class SubgraphTest(TestCase):
    '''
        Note:
            For DGLGraph (and subgraph), node and edge ids are always continuous integers starting from 0
    '''
    def test_edge_subgraph(self):
        '''
         2种等价形式：
            dgl.edge_subgraph(g, edges, relabel_nodes=True, store_ids=True)
            g.edge_subgraph(edges, relabel_nodes=True, store_ids=True)
            - edges: eid_tensor/eid_mask or {etype:eid_tensor/eid_mask}
            - relabel_nodes:  if True, remove isolated nodes, relabel nodes and store original Id in g.ndata[dgl.NID]
            - store_ids: if True, store original edges IDs in g.edata[dgl.EID]
        '''

        # homegeneous graph: 0-1-2-3-4-5
        g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]))

        # two ways to select subgraph
        eids = [0, 4]
        sg1 = g.edge_subgraph(eids)

        eid_mask = torch.tensor([True, False, False, False, True])  # choose edges [0, 4]
        sg2 = g.edge_subgraph(eid_mask)
        assert torch.equal(sg1.nodes(), sg2.nodes())
        assert torch.equal(sg1.edges()[0], sg2.edges()[0]) # 边的u相同
        assert torch.equal(sg1.edges()[1], sg2.edges()[1]) # 边的v相同

        # relabel_nodes option
        # relabel node
        sub_g = g.edge_subgraph([0, 4], relabel_nodes=True)
        new_eids = sub_g.edges(form='eid')
        old_eids = sub_g.edata[dgl.EID]
        new_nids = sub_g.nodes()
        old_nids = sub_g.ndata[dgl.NID]

        assert torch.equal(new_eids, torch.tensor([0, 1]))
        assert torch.equal(old_eids, torch.tensor([0, 4]))
        assert torch.equal(new_nids, torch.tensor([0, 1, 2, 3]))
        assert torch.equal(old_nids.sort().values, torch.tensor([0, 1, 4, 5]))

        # not relabel nodes: keep all nodes
        sub_g = g.edge_subgraph([0, 4], relabel_nodes=False)
        new_eids = sub_g.edges(form='eid')
        old_eids = sub_g.edata[dgl.EID]
        nids = sub_g.nodes()

        assert torch.equal(new_eids, torch.tensor([0, 1]))
        assert torch.equal(old_eids, torch.tensor([0, 4]))
        assert torch.equal(nids, torch.tensor([0, 1, 2, 3, 4, 5]))

        #heterogeneous graph:
        # user  game
        #
        g = dgl.heterograph({
            ('user', 'plays', 'game'): ([0, 1, 2], [2, 1, 0]),
            ('user', 'follows', 'user'): ([1, 7], [2, 8])
        })
        sub_g = dgl.edge_subgraph(g, {('user', 'plays', 'game'): [1, 2],
                                      ('user', 'follows', 'user'): [1]})

        old_nids = {
            'user' :  sub_g.ndata[dgl.NID]['user'],
            'game' :  sub_g.ndata[dgl.NID]['game']
        }
        old_eids = {
            'plays' : sub_g.edata[dgl.EID][('user', 'plays', 'game')],
            'follows' : sub_g.edata[dgl.EID][('user', 'follows', 'user')]
        }
        assert torch.equal(old_nids['user'].sort().values,  torch.tensor([1, 2, 7, 8]))
        assert torch.equal(old_nids['game'].sort().values,  torch.tensor([0, 1]))
        assert torch.equal(old_eids['plays'].sort().values, torch.tensor([1, 2]))
        assert torch.equal(old_eids['follows'].sort().values, torch.tensor([1]))

        # todo: use g.filter_edges to get eids, and dgl.edge_subgraph to extract subgraph

    def test_node_subgraph(self):
        '''
            2种等价形式：
            dgl.node_subgraph(g, nodes, relabel_nodes=True, store_ids=True)
            g.node_subgraph(nodes, relabel_nodes=True, store_ids=True) -- 0.7.x还没有
                - nodes: node_tensor/nid_mask or {etype:node_tensor/nid_mask}
                - relabel_nodes:  if True,  relabel nodes and store original Id in g.ndata[dgl.NID] (not remove isolated nodes)
                - store_ids: if True, store original edges IDs in g.edata[dgl.EID]

            (另有单侧的子图，用法类似： dgl.in_subgraph(g, nodes,..), dgl.out_subgraph(g, nodes,..))
        '''

        # homegeneous graph: 0-1-2-3-4-5
        g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]))

        # two ways to select subgraph
        nids = torch.tensor([0, 1, 4])
        sg1 = dgl.node_subgraph(g, nids)

        nid_mask = torch.tensor([True, True, False, False, True, False])  # choose edges [0, 4]
        sg2 = dgl.node_subgraph(g, nid_mask)
        assert torch.equal(sg1.nodes(), sg2.nodes())
        assert torch.equal(sg1.edges()[0], sg2.edges()[0]) # 边的u相同
        assert torch.equal(sg1.edges()[1], sg2.edges()[1]) # 边的v相同

        assert torch.equal(sg1.nodes(), torch.tensor([0, 1, 2]))
        assert torch.equal(sg1.ndata[dgl.NID].sort().values, nids)

        # todo: use g.filter_nodes to get nodes, and dgl.node_subgraph to extract subgraph

    def test_node_type_subgraph(self):
        '''
            dgl.node_type_subgraph(g, ntypes): 子图包含ntypes类型的所有节点， 两端类型都在etypes的边，以及features
        '''
        g = dgl.heterograph({
            ('user', 'plays', 'game'): ([0, 1, 2], [2, 1, 0]),
            ('user', 'follows', 'user'): ([1, 7], [2, 8])
        })

        sub_g = g.node_type_subgraph(['user'])
        assert sub_g.num_nodes('user') == 9
        assert sub_g.num_edges('follows') == 2

    def test_edge_type_subgraph(self):
        '''
            dgl.edge_type_subgraph(g, etypes): 子图包含
                - etypes类型的所有边
                - 以及有关联ntypes的所有节点
                - 边与节点上的特征
        '''

        g = dgl.heterograph({
            ('user', 'plays', 'game'): ([0, 1, 2], [2, 1, 0]),
            ('user', 'follows', 'user'): ([1, 7], [2, 8])
        })
        g.edges['plays'].data['h'] = torch.tensor([[0.], [1.], [2.]])

        sub_g = g.edge_type_subgraph(['plays'])
        assert sub_g.num_nodes('user') == 9
        assert sub_g.num_edges('plays') == 3
        assert torch.equal(sub_g.edata['h'], torch.tensor([[0.], [1.], [2.]]))

    # todo: 完成测试
    def test_slice_subgraph(self):
        '''
            g[srctype, etype, dsttype]:
            说明： 获取符合匹配条件的子图
            参考：https://docs.dgl.ai/en/0.7.x/generated/dgl.DGLGraph.__getitem__.html
        '''

    def test_compact_graphs(self):
        '''
            dgl.compact_graphs(graphs, always_preserve=None, copy_ndata=True, copy_edata=True)
                作用：去除在所有graphs中无连边的节点， 返回新的graphs
                返回值：new_graphs
                参数:
                    - graphs: 所有图{ntype:node_num}需要完全相同
        '''

        g1 = dgl.heterograph({('user', 'plays', 'game'): ([1, 3], [3, 5])},
                             {'user': 20, 'game': 10})

        g2 = dgl.heterograph({('user', 'plays', 'game'): ([1, 6], [6, 8])},
                             {'user': 20, 'game': 10})

        # 单图
        new_g = dgl.compact_graphs(g1)
        assert new_g.num_nodes('user') == 2
        assert torch.equal(new_g.ndata[dgl.NID]['user'], torch.tensor([1, 3]))

        # 多图
        new_g1, new_g2 = dgl.compact_graphs([g1, g2])
        assert new_g1.num_nodes('user') == new_g2.num_nodes('user') == 3
        assert torch.equal(new_g1.ndata[dgl.NID]['user'], torch.tensor([1, 3, 6]))

    def test_sampling_subgraph(self):
        '''
        种子节点的采样子图：
            dgl.sampling.sample_neighbors(g, nodes, fanout, edge_dir='in', prob=None, ..)
            作用： 以nid_tensor为种子采样边，返回子图(所有节点+采样的边)
            参数：
                - nodes:  nid_tensor or {ntype:nid_tensor}
                - fanout: int or {etype:int}, 一个节点采样边的数量

            https://docs.dgl.ai/en/0.7.x/generated/dgl.sampling.sample_neighbors.html


        '''
        g = dgl.graph(([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]))
        g.edata['prob'] = torch.FloatTensor([0., 1., 0., 1., 1.])

        sub_g = dgl.sampling.sample_neighbors(g, [0], fanout=3, prob='prob')
        assert sub_g.num_nodes() == 6
        assert sub_g.num_edges() == 3
        assert torch.equal(sub_g.edata[dgl.EID].sort().values, torch.tensor([1, 3, 4]))

    # todo: impl & proper placement
    def test_node_edge_subset(self):
        '''
            获取符合特点条件的

            g.filter_nodes(predicate, ntype=None)
            参数:
                - predicate: nodes-->bool_tensor
            返回：nid_tensor

            g.filter_edges(predicate, etype=None)
            参数:
                - predicate: edges-->bool_tensor
            返回：eid_tensor

            参考：
            https://docs.dgl.ai/en/0.7.x/generated/dgl.DGLGraph.filter_nodes.html
            https://docs.dgl.ai/en/0.7.x/generated/dgl.DGLGraph.filter_edges.html
        '''
        pass

