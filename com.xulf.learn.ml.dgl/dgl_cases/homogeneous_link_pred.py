import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset('../ogbn-arxiv')
device = 'cpu'      # change to 'cuda' for GPU

graph, node_labels = dataset[0]
# Add reverse edges since ogbn-arxiv is unidirectional.
graph = dgl.add_reverse_edges(graph)
print(graph)
print(node_labels)

node_features = graph.ndata['feat']
node_labels = node_labels[:, 0]
num_features = node_features.shape[1]
num_classes = (node_labels.max() + 1).item()
print('Number of classes:', num_classes)

idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
valid_nids = idx_split['valid']
test_nids = idx_split['test']

negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
train_dataloader = dgl.dataloading.EdgeDataLoader(
    # The following arguments are specific to NodeDataLoader.
    graph,                                  # The graph
    torch.arange(graph.number_of_edges()),  # The edges to iterate over
    sampler,                                # The neighbor sampler
    negative_sampler=negative_sampler,      # The negative sampler
    device=device,                          # Put the MFGs on CPU or GPU
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)

input_nodes, pos_graph, neg_graph, mfgs = next(iter(train_dataloader))
print('Number of input nodes:', len(input_nodes))
print('Positive graph # nodes:', pos_graph.number_of_nodes(), '# edges:', pos_graph.number_of_edges())
print('Negative graph # nodes:', neg_graph.number_of_nodes(), '# edges:', neg_graph.number_of_edges())
print(mfgs)


import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class Model(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

model = Model(num_features, 128).to(device)


import dgl.function as fn
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

def inference(model, graph, node_features):
    with torch.no_grad():
        nodes = torch.arange(graph.number_of_nodes())
        sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
        train_dataloader = dgl.dataloading.NodeDataLoader(
            graph, torch.arange(graph.number_of_nodes()), sampler,
            batch_size=1024,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            device=device)

        result = []
        for input_nodes, output_nodes, mfgs in train_dataloader:
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']
            result.append(model(mfgs, inputs))
        return torch.cat(result)

import sklearn.metrics
def evaluate(emb, label, train_nids, valid_nids, test_nids):
    classifier = nn.Linear(emb.shape[1], num_classes).to(device)
    opt = torch.optim.LBFGS(classifier.parameters())

    def compute_loss():
        pred = classifier(emb[train_nids].to(device))
        loss = F.cross_entropy(pred, label[train_nids].to(device))
        return loss

    def closure():
        loss = compute_loss()
        opt.zero_grad()
        loss.backward()
        return loss

    prev_loss = float('inf')
    for i in range(1000):
        opt.step(closure)
        with torch.no_grad():
            loss = compute_loss().item()
            if np.abs(loss - prev_loss) < 1e-4:
                print('Converges at iteration', i)
                break
            else:
                prev_loss = loss

    with torch.no_grad():
        pred = classifier(emb.to(device)).cpu()
        label = label
        valid_acc = sklearn.metrics.accuracy_score(label[valid_nids].numpy(), pred[valid_nids].numpy().argmax(1))
        test_acc = sklearn.metrics.accuracy_score(label[test_nids].numpy(), pred[test_nids].numpy().argmax(1))
    return valid_acc, test_acc

# Training Loop
# ----------------------
model = Model(node_features.shape[1], 128).to(device)
predictor = DotPredictor().to(device)
opt = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()))

import tqdm
import sklearn.metrics

best_accuracy = 0
best_model_path = 'model.pt'
for epoch in range(1):
    with tqdm.tqdm(train_dataloader) as tq:
        for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
            # feature copy from CPU to GPU takes place here
            inputs = mfgs[0].srcdata['feat']

            outputs = model(mfgs, inputs)
            pos_score = predictor(pos_graph, outputs)
            neg_score = predictor(neg_graph, outputs)

            score = torch.cat([pos_score, neg_score])
            label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
            loss = F.binary_cross_entropy_with_logits(score, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

            if (step + 1) % 500 == 0:
                model.eval()
                emb = inference(model, graph, node_features)
                valid_acc, test_acc = evaluate(emb, node_labels, train_nids, valid_nids, test_nids)
                print('Epoch {} Validation Accuracy {} Test Accuracy {}'.format(epoch, valid_acc, test_acc))
                if best_accuracy < valid_acc:
                    best_accuracy = valid_acc
                    torch.save(model.state_dict(), best_model_path)
                model.train()

                # Note that this tutorial do not train the whole model to the end.
                break


# Evaluating Performance with Link Prediction (Optional)
# Positive pairs
test_pos_src, test_pos_dst = graph.edges()
# Negative pairs
test_neg_src = test_pos_src
test_neg_dst = torch.randint(0, graph.num_nodes(), (graph.num_edges(),))

test_src = torch.cat([test_pos_src, test_pos_dst])
test_dst = torch.cat([test_neg_src, test_neg_dst])
test_graph = dgl.graph((test_src, test_dst), num_nodes=graph.num_nodes())
test_graph.edata['label'] = torch.cat([torch.ones_like(test_pos_src), torch.zeros_like(test_neg_src)])

test_dataloader = dgl.dataloading.EdgeDataLoader(
    # The following arguments are specific to EdgeDataLoader.
    test_graph,                             # The graph to iterate edges over
    torch.arange(test_graph.number_of_edges()),  # The edges to iterate over
    sampler,                                # The neighbor sampler
    device=device,                          # Put the MFGs on CPU or GPU
    g_sampling=graph,                       # Graph to sample neighbors
    # The following arguments are inherited from PyTorch DataLoader.
    batch_size=1024,    # Batch size
    shuffle=True,       # Whether to shuffle the nodes for every epoch
    drop_last=False,    # Whether to drop the last incomplete batch
    num_workers=0       # Number of sampler processes
)

test_preds = []
test_labels = []

with tqdm.tqdm(test_dataloader) as tq, torch.no_grad():
    for step, (input_nodes, pair_graph, mfgs) in enumerate(tq):
        # feature copy from CPU to GPU takes place here
        inputs = mfgs[0].srcdata['feat']

        outputs = model(mfgs, inputs)
        test_preds.append(predictor(pair_graph, outputs))
        test_labels.append(pair_graph.edata['label'])

test_preds = torch.cat(test_preds).cpu().numpy()
test_labels = torch.cat(test_labels).cpu().numpy()
auc = sklearn.metrics.roc_auc_score(test_labels, test_preds)
print('Link Prediction AUC:', auc)
