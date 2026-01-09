import torch_geometric
from torch_geometric._datasets import  Planetoid

#########################
## Dataset preparation ##
#########################

root_path = '/home/longdpt/Documents/Long_AISDL/DeepLearning_PyTorch/05_GNN/data'

#---------
## Load the dataset
#---------

dataset = Planetoid(root=root_path, name="Cora")

'''
Planetoid is not a single dataset, but rather a collection of three citation network datasets, 
commonly used for benchmarking Graph Neural Networks (GNNs).
'''

#---------
## Dataset properties
#---------

print(dataset) # Cora()
print("number of graphs:\t\t", len(dataset))                      # 1 (has only one graph)
print("number of classes:\t\t", dataset.num_classes)              # 7 (has 7 different features)
print("number of node features:\t", dataset.num_node_features)    # 1433 (each node has 1433 features)
print("number of edge features:\t", dataset.num_edge_features)    # 0

#---------
## dataset._data
#---------

print(dataset._data)
print("\n")
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
'''
2,708 nodes (papers)
10,556 edges (citations)
1,433 features (words)
7 classes (research topics)
'''

print("edge_index:\t\t", dataset._data.edge_index.shape)
print(dataset._data.edge_index)
print("\n")
# edge_index:		 torch.Size([2, 10556])              10556 edges (citation relationships)
# tensor([[ 633, 1862, 2582,  ...,  598, 1473, 2706],    SOURCE NODES
#         [   0,    0,    0,  ..., 2707, 2707, 2707]])   TARGET NODES
# Example: 633 -> 0, 1862 -> 0

print("train_mask:\t\t", dataset._data.train_mask.shape)
print(dataset._data.train_mask)
print("\n")
# train_mask:		 torch.Size([2708])
# tensor([ True,  True,  True,  ..., False, False, False])
# True: this is from the training set
# False: this is NOT in the training set

print("X:\t\t", dataset._data.x.shape)
print(dataset._data.x)
print("\n")
# X:		 torch.Size([2708, 1433])
# tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         ...,
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.],
#         [0., 0., 0.,  ..., 0., 0., 0.]])
# Each row represents one node
# Each column represents one feature of a node

print("y:\t\t", dataset._data.y.shape)
print(dataset._data.y)
print("\n")
# y:		 torch.Size([2708])
# tensor([3, 4, 4,  ..., 3, 3, 3])
# The output label of each node (in this example, we have 7 different classes)

#----
## get the data
#----

data = dataset[0] # Since we have only one dataset, use [0] to get it out

################
## Simple GNN ##
################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv # GraphSAGE Convolution layer from PyTorch Geometric

'''
What is SAGEConv?
GraphSAGE (SAmple and aggreGatE) is a type of graph convolution that:
# Samples neighbors of each node
# Aggregates their features
# Combines with the node's own features

What happens inside SAGEConv:
For each node i:
# Gather features from neighbors: {x_j : j ∈ Neighbors(i)}
# Aggregate: h_neighbors = max(x_j for all neighbors j) (element-wise max)
# Combine: h_i = W * concat([x_i, h_neighbors])

############ Example #############

x_i = [1, 0, 1]

neighbor_1 = [1, 2, 1]
neighbor_2 = [3, 4, 2]
neighbor_3 = [8, 1, 0]

aggregated = [max(1, 3, 8),   # position 0
              max(2, 4, 1),   # position 1
              max(1, 2, 0)]   # position 2

aggregated = [8, 4, 2]  ✅

=> concat([x_i, aggregated]) = [1, 0, 1, 8, 4, 2]
```

---

## Visual Representation
```
Position:       0   1   2
              ┌───┬───┬───┐
Neighbor 1:   │ 1 │ 2 │ 1 │
              ├───┼───┼───┤
Neighbor 2:   │ 3 │ 4 │ 2 │
              ├───┼───┼───┤
Neighbor 3:   │ 8 │ 1 │ 0 │
              ├───┼───┼───┤
              │ ↓ │ ↓ │ ↓ │
              ├───┼───┼───┤
Max:          │ 8 │ 4 │ 2 │
              └───┴───┴───┘
              
Node's own features:     [1, 0, 1]
                             +
Aggregated neighbors:    [8, 4, 2]
                              ↓
Concatenated:            [1, 0, 1, 8, 4, 2]
                         └─────┘ └───────┘
                          self   neighbors
'''

#----
## build model
#----

class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = SAGEConv(
            in_channels=dataset.num_features, # 1433 (Cora)
            out_channels=dataset.num_classes, # 7 (Cora)
            aggr="max" # could be max, mean, add, ...
        )
        
    def forward(self):
        out = self.cnn(data.x, data.edge_index)
        out = F.log_softmax(out, dim=1)
        return out
    
###############
## Optimizer ##
###############

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
model, data = SimpleGNN().to(device), data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

##############
## Training ##
##############

from loguru import logger

best_val_acc = test_acc = 0

for epoch in range(1, 101, 1):
    #----TRAIN
    _ = model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
    
    #-----VAL - TEST
    _ = model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
        
    _, val_acc, tmp_test_acc = accs
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        
    if (epoch % 10 == 0) or (epoch == 1):
        logger.info("+"*50)
        logger.info(f"Epoch: {epoch}")
        logger.info(f"Val: {best_val_acc:.4f}")
        logger.info(f"Test: {test_acc:.4f}")
'''
2026-01-09 13:56:53.950 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:53.950 | INFO     | __main__:<module>:27 - Epoch: 1
2026-01-09 13:56:53.950 | INFO     | __main__:<module>:28 - Val: 0.4380
2026-01-09 13:56:53.951 | INFO     | __main__:<module>:29 - Test: 0.4330
2026-01-09 13:56:53.969 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:53.969 | INFO     | __main__:<module>:27 - Epoch: 10
2026-01-09 13:56:53.969 | INFO     | __main__:<module>:28 - Val: 0.7260
2026-01-09 13:56:53.969 | INFO     | __main__:<module>:29 - Test: 0.7180
2026-01-09 13:56:53.990 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:53.990 | INFO     | __main__:<module>:27 - Epoch: 20
2026-01-09 13:56:53.990 | INFO     | __main__:<module>:28 - Val: 0.7260
2026-01-09 13:56:53.991 | INFO     | __main__:<module>:29 - Test: 0.7180
2026-01-09 13:56:54.011 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.011 | INFO     | __main__:<module>:27 - Epoch: 30
2026-01-09 13:56:54.011 | INFO     | __main__:<module>:28 - Val: 0.7260
2026-01-09 13:56:54.011 | INFO     | __main__:<module>:29 - Test: 0.7180
2026-01-09 13:56:54.032 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.032 | INFO     | __main__:<module>:27 - Epoch: 40
2026-01-09 13:56:54.032 | INFO     | __main__:<module>:28 - Val: 0.7260
2026-01-09 13:56:54.033 | INFO     | __main__:<module>:29 - Test: 0.7180
2026-01-09 13:56:54.054 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.054 | INFO     | __main__:<module>:27 - Epoch: 50
2026-01-09 13:56:54.054 | INFO     | __main__:<module>:28 - Val: 0.7260
2026-01-09 13:56:54.055 | INFO     | __main__:<module>:29 - Test: 0.7180
2026-01-09 13:56:54.074 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.075 | INFO     | __main__:<module>:27 - Epoch: 60
2026-01-09 13:56:54.075 | INFO     | __main__:<module>:28 - Val: 0.7260
2026-01-09 13:56:54.075 | INFO     | __main__:<module>:29 - Test: 0.7180
2026-01-09 13:56:54.095 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.095 | INFO     | __main__:<module>:27 - Epoch: 70
2026-01-09 13:56:54.096 | INFO     | __main__:<module>:28 - Val: 0.7260
2026-01-09 13:56:54.096 | INFO     | __main__:<module>:29 - Test: 0.7180
2026-01-09 13:56:54.116 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.116 | INFO     | __main__:<module>:27 - Epoch: 80
2026-01-09 13:56:54.117 | INFO     | __main__:<module>:28 - Val: 0.7280
2026-01-09 13:56:54.117 | INFO     | __main__:<module>:29 - Test: 0.7100
2026-01-09 13:56:54.137 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.137 | INFO     | __main__:<module>:27 - Epoch: 90
2026-01-09 13:56:54.138 | INFO     | __main__:<module>:28 - Val: 0.7340
2026-01-09 13:56:54.138 | INFO     | __main__:<module>:29 - Test: 0.7160
2026-01-09 13:56:54.158 | INFO     | __main__:<module>:26 - ++++++++++++++++++++++++++++++++++++++++++++++++++
2026-01-09 13:56:54.158 | INFO     | __main__:<module>:27 - Epoch: 100
2026-01-09 13:56:54.158 | INFO     | __main__:<module>:28 - Val: 0.7340
2026-01-09 13:56:54.158 | INFO     | __main__:<module>:29 - Test: 0.7160
'''