# HubGNN

========

The code corresponds to article "HubGNN: Global Hub-Driven GNN with Theoretically Guided Edge Partitioning".

HubGNN is a novel framework designed to address the challenges of training GNNs on large-scale graphs by combining edge partitioning with global information sharing. While edge partitioning retains the full graph structure, it introduces challenges such as determining its impact on local learning accuracy and addressing the information loss caused by cut nodes across subgraphs. HubGNN tackles these issues by first identifying HDRF as the optimal edge partitioning algorithm through theoretical and experimental analysis. Additionally, it introduces a global hub module to learn and share global graph information, mitigating the limitations of cut nodes. Experiments on six datasets show that HubGNN outperforms baseline methods, achieving up to 7.2% accuracy improvement, demonstrating its effectiveness in optimizing both partitioning and information sharing.


### Experimental environment 
All the experiments are conducted on a machine with NVIDIA GeForce RTX 3090GPU (24GB GPU memory), Intel Xeon Silver 4214R CPU(12 cores, 2.40GHz), and 256GB of RAM.


### Requirements
The codebase is implemented in Python 3.8.12. package versions used for development are just below.
```
torch              1.9.0
ogb                1.3.5
dgl                0.9.1
numpy              1.21.2
pandas             2.0.3
torchmetrics       1.0.3
torch_geometric    2.1.0
```
### Datasets

Computer, Physics, Flickr and Reddit are obtained from the Deep Graph Librar ([DGL](https://www.dgl.ai/)). The ogbn-arxiv and ogbn-products datasets are obtained from the Open Graph Benchmark ([OGB](https://ogb.stanford.edu/)).

### Parameter settings 

The GNNs on all datasets are 2 layers and 128 hidden units. The dropout rate is 0.5. We use ADAM as the optimizer. The learning rate is 0.001 and the weight decay is 0.0005. In order to compare the experimental results fairly, the number of subgraphs was kept the same under the same dataset. Note that edge partitioning algorithms require the node IDs in the dataset to be continuous. Therefore, we preprocess the datasets by eliminating isolated nodes (i.e., nodes with no edges), and reassigning continuous node IDs.

### Examples
<p align="justify">
The following commands learn a neural network and score on the test set. </p>

```sh
$ python HubGNN/train_HubGNN.py
```
