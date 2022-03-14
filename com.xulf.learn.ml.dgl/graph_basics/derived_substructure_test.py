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

    def test_random_walk(self):
        '''
        游走路径：

            dgl.sampling.random_walk(g, nid_tensor, metapath=None, length=None, prob=None, restart_prob=None, return_eids=False)
            参数：
                - nid_tensor: 游走起点 (一个起点产生一条路径, 允许重复）
                - metapath: [etype1, etype2, .., etype_k], 游走路径（如果不指定，认为g是同构图)
                    - length： 不指定metapath时，才设置；路径长度
                - prob: str, name of edge feature storing walk probabiliy (unnormalized)
                            if None, 均匀游走
                - restart_prob: 提前结束概率，float or 1d_tensor with len(metapath)/length

            返回：
                - traces: 游走node路径,  (num_seeds, len(metapath)+1). 如果提前结束, 补 -1
                - ntype_ids: 路径上节点类型, len(metapath) + 1

            dgl.sampling.node2vec_random_walk(g, nodes, p, q, walk_length, prob=None, return_eids=False)
            说明: node2vec类型随机游走，仅支持同构图
            返回：traces
        '''

        g2 = dgl.heterograph({
            ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
            ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
            ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])
        })

        # 2跳游走，仅在view 类型边依概率提前结束
        traces, ntype_ids = dgl.sampling.random_walk(g2, torch.tensor([0, 1, 2, 0]),
                                 metapath=['follow', 'view', 'viewed-by'] * 2,
                                 restart_prob=torch.FloatTensor([0, 0.5, 0, 0, 0.5, 0]))
        print(traces)
