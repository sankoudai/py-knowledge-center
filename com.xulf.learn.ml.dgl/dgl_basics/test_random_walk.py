import dgl
import torch

g1 = dgl.graph(([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]))
g1.edata['p'] = torch.FloatTensor([1, 0, 1, 1, 1])     # disallow going from 1 to 2
dgl.sampling.random_walk(g1, [0, 1, 2, 0], length=4, prob='p')
