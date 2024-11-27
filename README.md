# HubGNN

========

The code corresponds to article "HubGNN: Global Hub-Driven GNN with Theoretically Guided Edge Partitioning".

HubGNN is a novel framework designed to address the challenges of training GNNs on large-scale graphs by combining edge partitioning with global information sharing. While edge partitioning retains the full graph structure, it introduces challenges such as determining its impact on local learning accuracy and addressing the information loss caused by cut nodes across subgraphs. HubGNN tackles these issues by first identifying HDRF as the optimal edge partitioning algorithm through theoretical and experimental analysis. Additionally, it introduces a global hub module to learn and share global graph information, mitigating the limitations of cut nodes. Experiments on six datasets show that HubGNN outperforms baseline methods, achieving up to 7.2% accuracy improvement, demonstrating its effectiveness in optimizing both partitioning and information sharing.
