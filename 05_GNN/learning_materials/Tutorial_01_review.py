'''
Officical website: https://antoniolonga.github.io/Pytorch_geometric_tutorials/index.html

Tutorial_01:
https://www.youtube.com/watch?v=JtDgmmQ60x8&list=PLGMXrbDNfqTzqxB1IGgimuhtfAhGd8lHF

#######################################

The tutorial described in the sources is an introduction to Geometric Deep Learning (GDL) and the PyTorch Geometric framework, 
presented by Antonio Longa and Gabriele Santin [1, 2]. 

According to the sources, GDL is becoming essential in specialized areas like biology, network science, and recommendation systems 
where data does not fit traditional grid-like structures [1, 3].

The sources explain that conventional deep learning methods for images or speech cannot be applied to graphs 
because graphs lack a fixed structure and are not invariant to node ordering [4, 5].
For example, if the nodes of a graph are reordered, the resulting adjacency matrix changes entirely 
even if the graph itself remains identical, which confuses standard neural networks [6]. 
Furthermore, standard networks cannot easily handle the varying sizes of graphs or the addition of new nodes and edges [5, 6].

The core mechanism of GNNs involves the "computation graph," 
where each node generates an embedding by aggregating information from its immediate neighbors [7]. 
This process requires an "order-invariant aggregator," such as a sum or mean, to ensure the calculation is the same 
regardless of the order in which neighbors are processed [8]. 

Every node in a graph has its own unique computation graph based on its neighbors, 
but the trainable weights and biases are shared across the entire network to allow for generalization [8, 9]. 
The depth of the model determines how many "hops" or layers of neighbors are considered, 
though the sources suggest that looking too far into the network can collect redundant or irrelevant data [10, 11].

A specific model discussed is GraphSAGE, 
which uses general aggregation functions (like max-pooling) and concatenates a node's own previous state 
with the aggregated neighborhood data rather than simply summing them [12, 13]. 

The practical portion of the tutorial demonstrates these concepts using the Cora citation dataset, 
where papers are nodes and citations are edges [14]. 
The implementation involves creating a model class in PyTorch Geometric, using SAGEConv layers, 
and optimizing the network to classify nodes into one of seven categories [14-16].

Finally, the sources mention that GNNs are versatile enough to handle node classification, regression, and edge classification [17, 18]. 
To manage issues like self-loops, practitioners can treat self-edges as distinct connection types or limit the depth of the graph unrolling [18, 19].

The following analogy is not from the sources: To understand neighborhood aggregation, 
imagine a social gathering where you want to know the "vibe" of the party; 
you don't look at the whole room at once, but instead talk to your immediate friends, who each pass on the mood of the people they just spoke to, 
allowing information to flow to you one "hop" at a time.
'''