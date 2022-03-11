from unittest import TestCase
import dgl
import torch

class GraphPropTest(TestCase):
    def test_graph_type(self):
        '''
            图分类：
                1. 节点/边类型： 同构 vs 异构
                2. 图结构：二部图 vs 非二部图
        :return:
        '''
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

    def test_features(self):
        '''
            节点、边ID: g.nodes(ntype=None),  g.edges(form='uv', order='eid',  etype=None)
            节点/边特征：
                - g.nodes[ntype].data[feat] = tensor or g.ndata[ntype] = {ntype:tensor}
                - g.edges[etype].data[feat] = tensor or g.edata[etype] = {etype:tensor}
        '''
        g = dgl.heterograph({
            ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        })

        # 获取node,edge ID:
        assert g.nodes('user').tolist() == [0, 1, 2, 3, 4]

        assert torch.equal(g.edges(etype='plays')[0], torch.tensor([3, 4]))
        assert torch.equal(g.edges(etype='plays')[1], torch.tensor([5, 6]))
        assert g.edges(form='eid', etype='plays').tolist() == [0, 1]

        # get&set node features
        game_h, user_h = torch.zeros(7, 1), torch.zeros(5, 1)
        feat_dict = {'game': game_h, 'user': user_h}
        # g.nodes[ntype].data[feat] = tensor
        g.nodes['game'].data['h'] = game_h
        assert torch.equal(g.nodes['game'].data['h'],  game_h)
        assert torch.equal(g.ndata['h']['game'], game_h)

        # g.ndata[ntype] = {ntype:tensor}
        g.ndata['h'] = feat_dict
        assert torch.equal(g.ndata['h']['game'],  game_h)
        assert torch.equal(g.nodes['game'].data['h'],  game_h)

        # get&set edge features
        # g.edges[etype].data[feat] = tensor
        play_h = torch.ones(2, 1)
        g.edges['plays'].data['h'] = play_h
        assert torch.equal(g.edges['plays'].data['h'], play_h)
        assert torch.equal(g.edges[('user', 'plays', 'game')].data['h'], play_h)
        assert torch.equal(g.edata['h'][('user', 'plays', 'game')], play_h)

        # g.edata[etype] = {etype:tensor}
        play_h = torch.zeros(2, 1)
        g.edata['h'] = {'plays':play_h}
        assert torch.equal(g.edges['plays'].data['h'], play_h)
        assert torch.equal(g.edata['h'][('user', 'plays', 'game')], play_h)

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

    def test_adjacency_info(self):
        '''
            邻接信息： 给定tensor - u, v
            1. 连接边： 是否有边 - g.has_edges_between(u, v, etype=None)， 连接边id - g.edge_ids(u, v, etype=None)
            2. 入度出度：g.in_degrees(v, etype=None),  g.out_degrees(u, etype=None)
            3. 入边出边： 所有边放一起返回 g.in_edges(v, etype=None), g.out_edges(u, etype=None)
            4. 邻接节点: g.successors(u, etype=None), g.predecessors(v, etype=None)
            5. 获取邻接矩阵：g.adj(fmt=etype=None)
        '''
        g = dgl.heterograph({
            ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ('user', 'follows', 'game'): (torch.tensor([0, 1, 0]), torch.tensor([1, 2, 3])),
            ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        })

        #连接边信息
        user_u, game_v = torch.tensor([1, 2, 3]), torch.tensor([2, 3, 3])

        edge_mask = g.has_edges_between(user_u, game_v, etype='plays')
        user_u, game_v = user_u[edge_mask], game_v[edge_mask]
        edge_ids = g.edge_ids(user_u, game_v, etype='plays')

        assert edge_mask.tolist() == [True, False, True]
        assert user_u.tolist() == [1, 3]
        assert game_v.tolist() == [2, 3]
        assert edge_ids.tolist() == [0, 1]

        # 入度出度
        user_nodes = torch.tensor([0,1])
        in_degrees = g.in_degrees(user_nodes, etype=('user', 'follows', 'user'))
        out_degrees = g.out_degrees(user_nodes, etype=('user', 'follows', 'game'))
        assert in_degrees.tolist() == [0, 1]
        assert out_degrees.tolist() == [2, 1]

        # 入边出边
        user_nodes = torch.tensor([0,1])
        out_edges = g.out_edges(user_nodes, etype=('user', 'follows', 'game'), form='eid')
        assert out_edges.tolist() == [0, 2, 1]

        #邻接节点: g.predecessors 只能查看单一节点的前序节点
        pre_nodes = g.predecessors(1, etype=('user', 'follows', 'user'))
        post_nodes = g.successors(1, etype=('user', 'follows', 'user'))
        assert pre_nodes.tolist() == [0]
        assert post_nodes.tolist() == [2]

        #邻接矩阵
        m = g.adj(etype=('user', 'plays', 'game'))
        assert m[1, 2] == 1
        assert m[3, 3] == 1
        assert m[0, 2] == 0
        assert torch.sparse.sum(m)==2