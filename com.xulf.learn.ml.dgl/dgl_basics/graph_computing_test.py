import unittest
import dgl
import dgl.function as fn
import torch
import math
from dgl.nn.functional import edge_softmax


class GraphComputeTest(unittest.TestCase):
    '''
        三类计算:
        - 依赖单边特征的计算、依赖单节点特征的计算
            逻辑上可以使用简单的tensor operations与特征赋值来达到， dgl也提供了声明式方法g.apply_nodes, g.apply_edges
        - 计算依赖图连接关系的计算
            使用imperative方式表达很复杂， dgl提供了declarative methods来完成: g.update_all, g.multi_update_all
    '''

    def test_local_computation(self):
        '''
            g.apply_nodes(node_udf, ntype=None)
            说明：
                node_udf: nodes-->{feat:node_feat_tensor}, 参考 https://docs.dgl.ai/en/0.7.x/api/python/udf.html#apiudf
                    - nodes：NodeBatch类型
                        - nodes.data[feat]: a view of the node features
                        - nodes.mailbox[msg_name]: a view of the messages (仅仅在g.update_all中使用)

            g.apply_edges(edge_udf, etype=None)
            说明：
                edge_udf: edges-->{feat:edge_feat_tensor}, 参考 https://docs.dgl.ai/en/0.7.x/api/python/udf.html#apiudf
                    -edges: EdgeBatch类型
                        - edges.src[feat]: src node features
                        - edge.dst[feat]:  dsg node features
                        - edge.data[feat]: edge features
        '''

        g = dgl.heterograph({('user', 'follows', 'user'): ([0, 1], [1, 2])})

        # apply_nodes
        g.nodes['user'].data['h'] = torch.ones(3, 5)
        g.apply_nodes(lambda nodes: {'h': nodes.data['h'] * 2}, ntype='user')
        assert torch.equal(g.nodes['user'].data['h'], torch.ones(3, 5) * 2)

        # apply_nodes
        # edge feat compute
        g.edges[('user', 'follows', 'user')].data['h'] = torch.ones(2, 5)
        g.apply_edges(lambda edges: {'h': edges.data['h'] * 2})
        assert torch.equal(g.edges[('user', 'follows', 'user')].data['h'], torch.ones(2, 5) * 2)

        # use node features
        g.nodes['user'].data['h'] = torch.ones(3, 5)
        g.apply_edges(fn.u_add_v('h', 'h', 'x'))
        assert torch.equal(g.edges[('user', 'follows', 'user')].data['x'], torch.ones(2, 5) * 2)

    def test_propagation_computation(self):
        '''
            g.update_all(message_func, reduce_func, apply_node_func=None, etype=None)
            作用： 沿每条边生成dst节点消息m=message_func(x_u, x_v, x_e)， dst节点做聚合feat=reduce_func({m}, x_v)，将feat设置到目标节点v上
            参数：
                message_func： edge_udf,
                reduce_func:  一类特殊的node_udf， 其nodes.mailbox(msg_name)包含了接受到到的message (message_func生成)
                apply_node_func: node_udf, message reducec执行后进一步设置节点特征；一般设置为None，因可通过
            备注：df.function中的实现了常见的message_func， reduce_func

            g.multi_update_all(etype_dict, cross_reducer, apply_node_func=None)
            作用：沿每类边进行消息生成、聚合(reduce); 然后对一个dst node，使用cross_reducer做跨etype信息聚合
            参数：
                etype_dict:  { etype: (message_func, reduce_func, [apply_node_func])}
                cross_reducer: tensor_list-->tensor的聚合函数 （内置 'sum', 'mean', 'min', 'max', 'stack')
            备注：逻辑上，可以使用多个g.update_all代替
        '''

        #g.update_all
        # 完全二叉树图: 1,2->0, 3,4->1, 5,6->2
        g = dgl.heterograph({('user', 'follows', 'user'): ([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2])})
        g.nodes['user'].data['h'] = torch.ones(7, 1)
        g.edges['follows'].data['w'] = torch.tensor([[1.0], [1.0], [0.5], [0.5], [0.1], [0.1]])
        print(g.edges())

        # udf way: msg_func->edge_udf, reduce_func->sum_udf
        def edge_msg_udf(edges):
            return {'m': (edges.src['h'] + 2.0 * edges.dst['h']) * edges.data['w']}

        def sum_udf(dst_nodes):
            return {'x' : dst_nodes.mailbox['m'].sum(axis=1)}

        g['follows'].update_all(edge_msg_udf, sum_udf, etype='follows')
        assert torch.equal(g.nodes['user'].data['x'], torch.tensor([[6.],[3.],[0.6],[0.],[0.],[0.],[0.]]))

        # dgl.function:
        #   - fn.copy_u('h', 'm') - msg为src节点的h特征，存到dst节点mailbox['m']中
        #   - fn.sum('m', 'x') - 将dst.mailbox['m']加和， 设置到dst的特征x上
        g['follows'].update_all(fn.copy_u('h', 'm'), fn.sum('m', 'x'), etype='follows')
        assert torch.equal(g.nodes['user'].data['x'],  torch.tensor([[2.], [2.], [2.], [0.], [0.], [0.], [0.]]))

        #g.multi_update_all
        g = dgl.heterograph({
            ('user', 'follows', 'user'): ([0, 1], [1, 2]),
            ('game', 'attracts', 'user'): ([0], [1])
        })
        g.nodes['user'].data['h'] = torch.ones(3, 1)
        g.nodes['game'].data['h'] = torch.tensor([[1.]])

        g.multi_update_all(
            {'follows': (fn.copy_u('h', 'm'), fn.sum('m', 'h')),
             'attracts': (fn.copy_u('h', 'm'), fn.sum('m', 'h'))},
            "sum")
        assert torch.equal(g.nodes['user'].data['h'], torch.tensor([[0.], [2.], [1.]]))

        g.nodes['user'].data['h'] = torch.ones(3, 1)
        g.nodes['game'].data['h'] = torch.tensor([[1.]])
        def cross_sum(feat_list):
            return torch.sum(torch.stack(feat_list, dim=0), dim=0) if len(feat_list) > 1 else feat_list[0]
        g.multi_update_all(
            {'follows': (fn.copy_u('h', 'm'), fn.sum('m', 'h')),
             'attracts': (fn.copy_u('h', 'm'), fn.sum('m', 'h'))},
            cross_sum)
        assert torch.equal(g.nodes['user'].data['h'], torch.tensor([[0.], [2.], [1.]]))


class GraphComputeUsageTest(unittest.TestCase):
    def test_edge_softmax(self):
        '''
            对边：u->v
                𝑎ij=exp(𝑧𝑖𝑗)/∑i∈N(j)exp(𝑧𝑖𝑗)
        '''
        # 星形图
        g = dgl.graph(([1, 2, 3], [0, 0, 0]))
        edge_logits = torch.tensor([[math.log(1)], [math.log(4)], [math.log(5)]])

        # 自定义edge_softmax
        def edge_softmax_func(g, edge_logits):
            # dst节点计算加和, 然后对边特征归一
            # g的边、节点特征不会改变！
            with g.local_scope():
                g.edata['exp_z'] = torch.exp(edge_logits)
                g.update_all(fn.copy_e('exp_z', 'h'), fn.sum('h', 'sum_exp_z'))
                g.apply_edges(lambda edges: {'a':edges.data['exp_z'] / edges.dst['sum_exp_z']})
                return g.edata['a']

        assert torch.equal(edge_softmax_func(g, edge_logits), torch.tensor([[0.1], [0.4] ,[0.5]]))
        assert torch.allclose(edge_softmax(g, edge_logits), torch.tensor([[0.1], [0.4] ,[0.5]]))


