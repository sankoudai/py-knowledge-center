import unittest
import dgl
from dgl import function as fn
import torch

class MiscTest(unittest.TestCase):
    # todo: impl & proper placement
    def test_filter(self):
        '''
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

    def test_local_scope(self):
        '''
            with g.local_scope():
                xx

            作用：a local scope context for the graph
                - out-place mutation to the feature data will not reflect to the original graph: =
                - Inplace operations do reflect to the original graph: +=, -=, *=, /=
                (update_all, apply_edges，apply_nodes 也不会改变节点/边特征！！）
            说明：常用于forward方法、 在图上的计算函数， 好处是不会影响图的边/节点特征
        '''

        g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        g.ndata['h'] = torch.zeros(3, 1)
        g.edata['w'] = torch.ones(2, 1)

        # out-place mutation
        with g.local_scope():
            g.ndata['h'] = g.ndata['h'] + 1
            assert torch.equal(g.ndata['h'], torch.ones(3, 1))
        assert torch.equal(g.ndata['h'], torch.zeros(3, 1))

        with g.local_scope():
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'h'))
        assert torch.equal(g.ndata['h'], torch.zeros(3, 1))


    # in-place mutation
        with g.local_scope():
            g.ndata['h'] +=  1
            assert torch.equal(g.ndata['h'], torch.ones(3, 1))
        assert torch.equal(g.ndata['h'], torch.ones(3, 1))