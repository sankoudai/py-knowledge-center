from unittest import TestCase
import dgl
import torch

class GraphPropTest(TestCase):
    def test_graph_type(self):
        # 同构图（一种节点与边） vs 异构图
        g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))
        assert g.is_homogeneous == True

        g = dgl.heterograph({('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))})
        assert g.is_homogeneous == False

        # 是否是二部图：节点类型可拆分为源节点类型、 汇节点类型的图(即一个节点类型，要么都是src，要么都是dst)
        g = dgl.heterograph({
            ('user', 'follow_game', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
            ('user', 'follow_user', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        })
        assert g.is_unibipartite == False

        g = dgl.heterograph({
            ('user', 'follow_game', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
            ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        })
        assert g.is_unibipartite == True


    def test_metagraph(self):
        '''
            两种方式获取g的元信息:
            1. 使用g的方法：ntypes, etypes,  canonical_etypes
            2. 获取g.metagraph(): 通过meta_g的节点、边信息
        '''
        g = dgl.heterograph({
            ('user', 'follow_game', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
            ('user', 'follow_user', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        })

        # node types & edge types & canonical edge types
        assert g.ntypes == ['game', 'user']
        assert g.etypes == ['follow_game', 'follow_user', 'plays']
        assert g.canonical_etypes == [
            ('user', 'follow_game', 'game'),
            ('user', 'follow_user', 'user'),
            ('user', 'plays', 'game')
        ]

        # metagraph: which is of type networkx.MultiDiGraph
        meta_g = g.metagraph()
        assert list(meta_g.nodes()) == ['user', 'game']
        assert list(meta_g.edges()) == [('user', 'game'), ('user', 'game'), ('user', 'user')]
        assert list(meta_g['user']['game']) == ['follow_game', 'plays']

    def test_graph_statistics(self):
        g = dgl.heterograph({
            ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        })

        # 节点与边数量
        assert g.num_nodes() == 12
        assert g.num_nodes('user') == 5

        assert g.num_edges() == 4
        assert g.num_edges('plays') == 2
        assert g.num_edges(('user', 'plays', 'game')) == 2