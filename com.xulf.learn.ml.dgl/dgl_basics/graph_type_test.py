import dgl
import torch
from unittest import TestCase

class TypeTest(TestCase):
    def test_homo_heter(self):
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


    def test_bipartie(self):
        '''
            二部图：
                定义：  节点类型可拆分为disjoint源节点类型、 汇节点类型的图(即一个节点类型，要么都是src，要么都是dst)
                判别方法：dgl.is_unibipartie
                特有方法：
                    -  获取src/dst nodeId: g.srcnodes(ntype=None), g.dstnodes(ntype=None)
                    -  获取src/dst ntypes: g.srctypes(), g.dsttypes()
                    -  获取src/dst 节点数: g.num_src_nodes(ntype=None)， g.num_dst_nodes(ntype=None)
                    -  单独get/set src/dst节点上的数据：g.srcdata[feat] = {ntype:tensor}, g.dstdata[feat]={ntype:tensor}
        '''
        # 同构图都非unibipartie
        g = dgl.graph(([0, 1, 2], [3, 4, 5]))
        assert g.is_unibipartite == False

        #如果一个ntype 同时包含src节点，dst节点， 则非unibipartie
        g = dgl.heterograph({
            ('user', 'follows', 'topic'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
            ('game', 'played-by', 'user'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        })
        assert g.is_unibipartite == False

        # bipartie的例子
        g = dgl.heterograph({
            ('user', 'follows', 'topic'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
            ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        })
        g.dstdata['h'] = {
            'topic':torch.ones(4, 1),
            'game' : torch.zeros(4, 1)
        }

        assert g.is_unibipartite == True
        assert torch.equal(g.dstnodes('game'), torch.tensor([0, 1, 2, 3]))
        assert g.dsttypes == ['game', 'topic']
        assert g.num_dst_nodes('game') == 4
        assert torch.equal(g.dstdata['h']['game'], torch.zeros(4, 1))

    def test_block(self):
        '''
            block: 一类特殊的unibipartie图，表示为DGLGraph的子类DGLBlock
            - 生成方式: dgl.to_block(g),  dgl.creat_block
            - 有二部图的属性与方法
            - 相比一般二部图的特殊处: src与dst节点类型， 可同名但其实是不同类型!! (抽象有点ugly)
        '''


        # dgl.to_block(g, dst_nodes=None, include_dst_in_src=True)
        # 返回： 生成一个block类型二部图block_g
        #       dst节点--参数指定
        #       src节点--有边与dst节点连接的节点 （可能与dst_nodes有交集， 交集节点与dst_nodes'看起来一样，实际不同', srcdata与dstdata对应数据不共享)
        #       边 -- 以dst_nodes为终点的边
        #       (说明：src/dst 在原图g中的id，存在g.srcdata[dgl.NID], g.dstdata[dgl.NID])
        # 参数说明：
        #   dst_nodes: nid_tensor or {ntype:nid_tensor},  nid_tensor必须包含所有ntype类型的dstnodes
        #   include_dst_in_src: if True， 将参数的dst_nodes也加入到srcnodes中，且srcnodes的前半部分与dst_nodes完全一样
        # 使用场景： neigbor sampling时用于生成mfg

        # 同构图
        g = dgl.graph(([1, 2], [3, 4]), num_nodes=8)
        g.ndata['h'] = torch.zeros(8, 1)

        dst_nodes = torch.tensor([4, 3])
        block_g = dgl.to_block(g, dst_nodes=dst_nodes)
        block_g.srcdata['h'] = torch.ones(4, 1)

        assert block_g.is_block
        assert block_g.is_unibipartite

        # 节点数： srcnodes两部分组成--与dst_nodes有连边的节点1,2，以及dst_nodes (当然会被relabel为从0开始!)
        assert block_g.num_dst_nodes() == 2
        assert block_g.num_src_nodes() == 4
        assert block_g.number_of_nodes() == 6
        assert torch.equal(block_g.dstdata[dgl.NID], dst_nodes)
        assert torch.equal(block_g.srcdata[dgl.NID], torch.tensor([4, 3, 1, 2]))

        #srcdata
        assert torch.equal(block_g.srcdata['h'], torch.ones(4, 1))
        assert torch.equal(block_g.dstdata['h'], torch.zeros(2, 1))

        # 异构图: 有关联的srcnodes都会包含进来
        g = dgl.heterograph({
            ('user', 'plays', 'game'): ([0, 1, 2], [2, 1, 0]),
            ('buyer', 'buys', 'game'): ([1, 7], [2, 8]),
        })
        dst_nodes = {'game':torch.tensor([0, 1, 2, 8])}
        block_g = dgl.to_block(g, dst_nodes)
        assert block_g.srctypes == ['buyer', 'game', 'user']
        assert block_g.num_src_nodes('buyer') == 2
        assert block_g.num_src_nodes('game') == 4
        assert block_g.num_src_nodes('user') == 3

        # 使用不多
        block = dgl.create_block({
            ('A', 'AB', 'B'): ([1, 2, 3], [2, 1, 0]),
            ('B', 'BA', 'A'): ([2, 1], [2, 3])},
            num_src_nodes={'A': 6, 'B': 5},
            num_dst_nodes={'A': 4, 'B': 3})
        # print(block.is_block)
        # print(block.is_unibipartite)
        # print(block.ntypes)
        # print(block.get_ntype_id_from_dst('A'))
        # print(block.get_ntype_id_from_src('A'))
        # print(block.get_ntype_id('A'))
