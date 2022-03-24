import unittest
import torch.nn as nn
import torch
import math
import dgl
from dgl.nn.functional import  edge_softmax
import dgl.function as fn
from dgl.utils import expand_as_pair

class GnnModuleTest(unittest.TestCase):
    '''
        单一边类型上的gnn module， 包含三种情况：
            - 同构图上的gnn module
            - 二部图上的gnn module (srtype != dsttype)
            - block上的gnn module (其实block是一种特殊的同构图， 又是一种特殊的二部图， 概念真不优美)
        一般gnn module都会同时兼容这种情况
    '''
    def test_module_on_homograph(self):
        '''
            一般图上的gnn module: 单一边类型上的gnn module
        '''
        class SimpleConv(nn.Module):
            def __init__(self):
                super(SimpleConv, self).__init__()

            def forward(self, g , edge_wght,  node_feat):
                with g.local_scope():
                    g.ndata['h'] = node_feat
                    norm_w = edge_softmax(g, edge_wght)
                    g.edata['norm_w'] = norm_w #只有一类边时，可以简化
                    g.update_all(fn.u_mul_e('h', 'norm_w', 'm'), fn.sum('m', 'h'))
                    return g.ndata['h'] + node_feat

        g = dgl.heterograph({('user', 'follows', 'user'): ([1, 2, 3], [0, 0, 0])})
        g.nodes['user'].data['h'] = torch.ones(4, 1)
        g.edges['follows'].data['w'] = torch.tensor([[math.log(1)], [math.log(4)], [math.log(5)]])

        conv = SimpleConv()
        assert torch.equal(
            conv(g, g.edges['follows'].data['w'], g.nodes['user'].data['h']),
            torch.tensor([[1.0], [0.], [0.], [0.]])
        )
    
    def test_module_on_bipartie(self):
        '''
            二部图上的gnn module
        '''
        class SimpleBipartieConv(nn.Module):
            def __init__(self, srcdim,  dstdim):
                super(SimpleBipartieConv, self).__init__()
                self.srcdim = srcdim
                self.dstdim = dstdim
                self.fc = nn.Linear(self.srcdim, self.dstdim)

                nn.init.xavier_uniform_(self.fc.weight) #初始化权重

            def forward(self, g , edge_wght, srcfeat, dstfeat):
                with g.local_scope():
                    #将srcfeat特征映射为dstdim维度
                    g.srcdata['h'] = self.fc(srcfeat)

                    norm_w = edge_softmax(g, edge_wght)
                    g.edata['norm_w'] = norm_w #只有一类边时，可以简化
                    g.update_all(fn.u_mul_e('h', 'norm_w', 'm'), fn.sum('m', 'h'))
                    return g.dstdata['h'] + dstfeat

        g = dgl.heterograph({('user', 'follows', 'topic'): ([1, 2, 3], [0, 0, 0])})

        g.srcdata['h'] = torch.ones(4, 2)
        g.dstdata['feat'] = torch.ones(1, 1)
        g.edges['follows'].data['w'] = torch.tensor([[math.log(1)], [math.log(4)], [math.log(5)]])

        conv = SimpleBipartieConv(2, 1)
        res = conv(g, g.edges['follows'].data['w'], g.srcdata['h'], g.dstdata['feat'])
        print(res)

    def test_mudule_on_block(self):
        '''
            block上的module(注这里是同构图block，对于两种节点的block，参考二部图上的module'
        '''
        class SimpleBlockConv(nn.Module):
            def __init__(self):
                super(SimpleBlockConv, self).__init__()

            def forward(self, g , edge_wght, srcfeat):
                dstfeat = srcfeat[:g.number_of_dst_nodes(),...]
                with g.local_scope():
                    #将srcfeat特征映射为dstdim维度
                    g.srcdata['h'] = srcfeat
                    norm_w = edge_softmax(g, edge_wght)
                    g.edata['norm_w'] = norm_w #只有一类边时，可以简化
                    g.update_all(fn.u_mul_e('h', 'norm_w', 'm'), fn.sum('m', 'h'))
                    return g.dstdata['h'] + dstfeat

        g = dgl.graph(([0, 1, 2], [4, 5, 6]), num_nodes=8)
        g.ndata['h'] = torch.ones(8, 1)

        dst_nodes = torch.tensor([4, 5, 6])
        block_g = dgl.to_block(g, dst_nodes=dst_nodes)

        conv = SimpleBlockConv()
        w= torch.tensor([[math.log(1)], [math.log(4)], [math.log(5)]])
        conv(block_g, w, block_g.srcdata['h'])

    def test_module_integrated(self):
        class SimpleConv(nn.Module):
            def __init__(self, srcdim, dstdim):
                super(SimpleConv, self).__init__()
                self.srcdim = srcdim
                self.dstdim = dstdim
                self.fc = nn.Linear(self.srcdim, self.dstdim)

                nn.init.xavier_uniform(self.fc.weight)

            def forward(self, g, edge_wgt, node_feats):
                '''
                    node_feats: 同构图为feat, 二部图时为(srcfeat, dstfeat), block为srcfeat
                '''
                with g.local_scope():
                    # src_feat, dst_feat
                    if isinstance(node_feats, tuple): #二部图
                        feat_src, feat_dst = node_feats
                    elif g.is_block: #block
                        feat_src, feat_dst = node_feats, node_feats[:g.number_of_dst_nodes(),...]
                    else: #同构图
                        feat_src, feat_dst = node_feats, node_feats

                    # 先传播后映射
                    g.srcdata['h'] = feat_src  #note 同构图的srcdata 就是ndata
                    g.edata['norm_w'] = edge_softmax(g, edge_wgt)
                    g.update_all(fn.u_mul_e('h', 'norm_w', 'm'), fn.sum('m', 'h'))
                    return self.fc(g.dstdata['h']) + feat_dst

        # block
        g = dgl.graph(([0, 1, 2], [4, 5, 6]), num_nodes=8)
        g.ndata['h'] = torch.zeros(8, 2)
        dst_nodes = torch.tensor([4, 5, 6])
        block_g = dgl.to_block(g, dst_nodes=dst_nodes)

        conv = SimpleConv(2, 2)
        w= torch.tensor([[math.log(1)], [math.log(4)], [math.log(5)]])
        new_h = conv(block_g, w, block_g.srcdata['h'])
        print(new_h)

        # bipartie
        g = dgl.heterograph({('user', 'follows', 'topic'): ([1, 2, 3], [0, 0, 0])})

        g.srcdata['h'] = torch.ones(4, 2)
        g.dstdata['feat'] = torch.ones(1, 1)
        g.edges['follows'].data['w'] = torch.tensor([[math.log(1)], [math.log(4)], [math.log(5)]])

        conv = SimpleConv(2, 1)
        new_h = conv(g, g.edges['follows'].data['w'], (g.srcdata['h'], g.dstdata['feat']))
        print(new_h)

        #homograph
        g = dgl.heterograph({('user', 'follows', 'user'): ([1, 2, 3], [0, 0, 0])})
        g.nodes['user'].data['h'] = torch.ones(4, 2)
        g.edges['follows'].data['w'] = torch.tensor([[math.log(1)], [math.log(4)], [math.log(5)]])

        conv = SimpleConv(2, 2)
        new_h =  conv(g, g.edges['follows'].data['w'], g.nodes['user'].data['h'])
        print(new_h)

