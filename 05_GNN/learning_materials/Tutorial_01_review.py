'''
Officical website: https://antoniolonga.github.io/Pytorch_geometric_tutorials/index.html

github: https://github.com/AntonioLonga/PytorchGeometricTutorial/tree/main?tab=readme-ov-file

Tutorial_01:
https://www.youtube.com/watch?v=JtDgmmQ60x8&list=PLGMXrbDNfqTzqxB1IGgimuhtfAhGd8lHF

#######################################

LECTURE OVERVIEW: INTRODUCTION TO GRAPH NEURAL NETWORKS (GNNs)

1. THE CORE CONCEPT
The lecture focuses on how to perform deep learning on graphs.
Unlike images (fixed grids) or text (sequences), graphs have an arbitrary structure and no fixed ordering of nodes. 
The goal of a GNN is to learn a numerical vector (an embedding) for every node that captures both its own features 
and its position/role within the graph structure.

2. THE COMPUTATION GRAPH
For any specific node (let's call it "Node A"), the GNN creates a computation graph based on its local neighborhood. 
- Layer 0: The initial input features of Node A and its neighbors.
- Layer 1: Node A aggregates information from its direct neighbors.
- Layer 2: Node A's neighbors aggregate information from their own neighbors (which are 2 steps away from Node A).
By stacking K layers, a node can "see" information from K steps away.

3. THE BASIC MATH OF MESSAGE PASSING
The process of updating a node's representation involves two main steps: Aggregation and Transformation.

The mathematical formula for a node "v" at layer "k" is:

h_v^(k) = SIGMA( W_k * AGGREGATE({ h_u^(k-1) for all u in Neighbors(v) }) + B_k * h_v^(k-1) )

Definitions:
- h_v^(k): The state (embedding) of node v at layer k.
- h_u^(k-1): The state of a neighbor node u from the previous layer.
- Neighbors(v): The set of nodes directly connected to v.
- W_k: A trainable weight matrix applied to neighbor information.
- B_k: A trainable weight matrix applied to the node's own previous state.
- SIGMA: A non-linear activation function (usually ReLU).
- AGGREGATE: A function like Sum, Mean, or Max that handles a variable number of neighbors.

4. GRAPH CONVOLUTIONAL NETWORKS (GCN) SPECIFIC MATH
In a standard GCN, the aggregation is often a simple normalized mean:

h_v^(k) = SIGMA( W_k * SUM[ h_u^(k-1) / SQRT(Degree(v) * Degree(u)) ] )

Here, Degree(v) is the number of edges connected to node v. 
This normalization ensures that nodes with many neighbors don't produce massive values that destabilize the neural network.

5. MATRIX FORM (FOR EFFICIENCY)
To compute this for all nodes at once, we use matrices:

H^(k) = SIGMA( INV(D) * A * H^(k-1) * W_k^T )

Definitions:
- H^(k): A matrix where each row is a node's embedding at layer k.
- A: The Adjacency Matrix (1 if nodes are connected, 0 otherwise).
- INV(D): The inverse of the Degree Matrix (used for averaging).
- W_k^T: The transposed weight matrix for the layer.

6. KEY TAKEAWAYS
- Local Neighborhoods: Nodes are defined by their neighbors.
- Weight Sharing: The same weights (W and B) are used for every node in the graph, making the model scale to very large graphs.
- Permutation Invariance: Because the AGGREGATE function (like Sum) doesn't care about the order of neighbors, the GNN produces the same result regardless of how nodes are indexed in the data.
'''