import dgl
import numpy as np
import torch as th
from dgl.nn import SAGEConv

# Case 1: Homogeneous graph
# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# feat = th.ones(6, 10)
# conv = SAGEConv(10, 2, 'pool')
# res = conv(g, feat)


u = [0, 1, 0, 0, 1]
v = [0, 1, 2, 3, 2]
g = dgl.heterograph({('_U', '_E', '_V'): (u, v)})
u_fea = th.rand(2, 5)
v_fea = th.rand(4, 10)
conv = SAGEConv((5, 10), 2, 'lstm')
res = conv(g, (u_fea, v_fea))
