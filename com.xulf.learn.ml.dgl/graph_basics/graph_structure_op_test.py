import dgl
import torch
from unittest import  TestCase

class StructureOpTest(TestCase):
    '''
        graph structure op:
            - generally returns a new graph.
            - original nid, eid stores in new_g.ndata[dgl.NID], g.edata[dgl.EID]
    '''

    def test_base_ops(self):
        '''
        添加/删除元素：
            dgl.add_nodes(g, num, data=None, ntype=None)
            dgl.remove_nodes(g, nids, ntype=None, store_ids=False)
            dgl.add_edges(g, u, v, data=None, etype=None)
            dgl.remove_edges(g, eids, etype=None, store_ids=False)
        '''

        g = dgl.heterograph({
            ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),  torch.tensor([0, 0, 1, 1])),
            ('developer', 'develops', 'game'): (torch.tensor([0, 1]), torch.tensor([0, 1]))
        })

        # add_nodes
        new_g = dgl.add_nodes(g, 2, ntype='user')
        assert g.num_nodes('user') == 3
        assert new_g.num_nodes('user') == 5

        # remove_nodes
        new_g = dgl.remove_nodes(g, [2], ntype='user', store_ids=True)
        print(new_g.ndata[dgl.NID])

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


