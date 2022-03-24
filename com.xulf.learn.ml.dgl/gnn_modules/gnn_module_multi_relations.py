import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch
from torch import nn
from unittest import TestCase

class MultiRelModuleTest(TestCase):
    def test_udf_module(self):
        '''
            异构图上消息传播机制：每种边分别做信息传播， 然后对给定类型节点的所有消息做聚合
            如下异构图的消息传递，有dgl的现成实现dgl.nn.HeteroGraphConv
        '''
        class HeteroGraphConv(nn.Module):
            def __init__(self, rel_mod_dict, agg_fn):
                '''
                    rel_mod_dict: {etype:single_rel_conv}
                    agg_fn: (tensors, ntype) --> tensor,   aggregate dst msgs from multiple relations (etypes)

                '''
                super(HeteroGraphConv, self).__init__()
                self.mods = nn.ModuleDict(rel_mod_dict)
                self.agg_fn = agg_fn

            def forward(self, g, nfeat_dict, rel_args_dict=None, rel_kwargs_dict=None):
                '''
                    node_feat_dict: {ntype: tensor}
                    rel_args_dict: {etype:args}
                    rel_kwargs_dict: {etype:kwargs}
                '''
                if rel_args_dict is None:
                    rel_args_dict = {}
                if rel_kwargs_dict is None:
                    rel_kwargs_dict = {}

                outputs = {nty : [] for nty in g.dsttypes}
                if g.is_block:
                    src_inputs = nfeat_dict
                    dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in nfeat_dict.items()}
                else:
                    src_inputs = dst_inputs = nfeat_dict

                #对每一种边，单独进行消息传递
                for srctype, etype, dsttype in g.canonical_etypes:
                    rel_graph = g[srctype, etype, dsttype]
                    if rel_graph.num_edges() == 0:
                        continue
                    if srctype not in src_inputs or dsttype not in dst_inputs:
                        continue

                    dstdata = self.mods[etype](
                        rel_graph,
                        (src_inputs[srctype], dst_inputs[dsttype]),
                        *rel_args_dict.get(etype, ()),
                        **rel_kwargs_dict.get(etype, {}))

                    outputs[dsttype].append(dstdata)

                # 对每一种边，聚合各种消息
                new_nfeat_dict = {}
                for nty, dstdatas in outputs.items():
                    if len(dstdatas) != 0:
                        new_nfeat_dict[nty] = self.agg_fn(dstdatas, nty)
                return new_nfeat_dict

        class SimpleConv(nn.Module):
            def __init__(self, srcdim, dstdim):
                super(SimpleConv, self).__init__()
                self.srcdim = srcdim
                self.dstdim = dstdim
                self.fc = nn.Linear(self.srcdim, self.dstdim)

                nn.init.xavier_uniform(self.fc.weight)

            def forward(self, g, node_feats, edge_wgt):
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

        g = dgl.heterograph({
            ('user', 'follows', 'user') : (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])),
            ('game', 'played-by', 'user'): (torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])),
            ('game', 'has-tag', 'tag'): (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1]))
        })
        g.nodes['user'].data['h'] = torch.ones([6, 2])
        g.nodes['game'].data['h'] = torch.rand([3, 3])
        g.nodes['tag'].data['h'] = torch.rand([2, 4])
        g.edges['played-by'].data['w'] = torch.rand([3, 1])

        rel_mod_dict = {
            'follows': dgl.nn.SAGEConv(2, 2, aggregator_type='mean'),
            'played-by':SimpleConv(3, 2),
            'has-tag':dgl.nn.SAGEConv((3, 4), 4,  aggregator_type='mean')
        }

        def user_aggregator(tensors, ntype):
            stacked = torch.stack(tensors, dim=0)
            if ntype=='user':
                return torch.sum(stacked, dim=0)
            else:
                return torch.mean(stacked, dim=0)


        conv = HeteroGraphConv(rel_mod_dict, user_aggregator)
        node_feat_dict = {'user':torch.ones([6, 2]),
                          'game':torch.rand([3, 3]),
                          'tag':torch.rand([2, 4])
                          }
        rel_args = {'played-by': [torch.rand([3, 1])]}
        new_feats = conv(g, node_feat_dict, rel_args_dict = rel_args)
        print(new_feats)

