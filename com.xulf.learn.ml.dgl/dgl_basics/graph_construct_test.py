import unittest
import dgl
import torch

class GraphConstructTest(unittest.TestCase):
    '''
     一般
        同构图： dgl.graph
        异构图： dgl.heterograph
    '''
    def test_graph(self):
        '''
            签名：dgl.graph(edge_data, num_nodes=None, itype=None ...)
            作用：生成一个同构图
            参数说明：
                edge_data: 标明边， 可以取(tensor_1d, tensor_1d)以及其他的稀疏表示形式
                num_nodes：node数， 从0开始
                itype: 存储节点的id类型， 一般为torch.int32, touch.int64
            参考：https://docs.dgl.ai/en/0.7.x/generated/dgl.graph.html
        '''

        # A graph with edges:  (2, 1), (3, 2), (4, 3),  nodes: 0~4
        src_nodes = torch.tensor([2, 3, 4])
        dst_nodes = torch.tensor([1, 2, 3])
        g = dgl.graph((src_nodes, dst_nodes))

        assert g.num_nodes() == 5
        assert g.num_edges() == 3

        # Same graph of node type int32, on first gpu
        g = dgl.graph((src_nodes, dst_nodes), idtype=torch.int32, device='cuda:0')
        assert g.num_nodes() == 5
        assert g.num_edges() == 3

    def test_heterograph(self):
        '''
            签名：dgl.heterograph(edge_dict, num_nodes_dict=None, itype=None, device=None)
            作用：生成一个异构图
            参数说明：
                edge_dict: 边数据
                    key: (src_type, edge_type, dst_type),
                    value: 可以取(tensor_1d, tensor_1d)以及其他的稀疏表示形式
                num_nodes_dict：{ntype:num}, （每个类型的节点id都是从0开始）
                itype: 存储节点的id类型， 一般为torch.int32, touch.int64
            参考：https://docs.dgl.ai/en/0.7.x/generated/dgl.heterograph.html
        '''

        # A heterograph with node:  4 user , 3 topic, 5 game; and edge
        edge_dict = {
            ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
            ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
        }
        g = dgl.heterograph(edge_dict)

        assert g.num_nodes() == 12
        assert g.num_nodes('user') == 4
        assert g.num_nodes('topic') == 3
        assert g.num_nodes('game') == 5

        assert g.num_edges() == 6
        assert g.num_edges(('user', 'follows', 'user')) == 2
        assert g.num_edges('plays') == 2

