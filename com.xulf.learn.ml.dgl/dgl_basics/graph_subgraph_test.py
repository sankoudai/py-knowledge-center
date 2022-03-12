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