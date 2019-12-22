# Fast-SimGNN

An attempt to beat SimGNN

## Overriew

## Abstract
In order to solve the pattern recognition problem such as classification and clustering, an effective and scalable measure of computing graph similarity and dissimilarity is necessary to be defined. Graph Edit Distance (GED) is a flexible measure of dissimilarity between graphs which is widely used in recent research.  It is defined from an optimal sequence of edit operations (edit path) transforming from a source graph into a target graph. The main advantage of GED is its flexibility and sensitivity to small differences between the input graphs. The main drawback is that it is hard to compute in practice. So graph similarity computation still remains challenging. Recent years have seen a surge in approaches that automatically learn to encode graph structure into low-dimensional embeddings, using techniques based on deep learning and nonlinear dimensionality reduction. Some scientists successfully leveraged learnable techniques to approximate the similarity between two graphs. However, existing deep learning based models are still time-consuming and not scalable for current applications. In this paper, we proposed a scalable graph similarity measuring framework based on graph decomposition. Our method combines both attention and partition mechanisms in the process of deep learning, whose time complexity reduced to quasilinear. In order to preserve the structure of the graph and the relation between individual nodes, our method jointly learns the graph structure and node-level embeddings simultaneously.  Experiments show that our approach achieves state-of-the-art performance in many situations.

## Reference
- [SimmGNN WSDM'19](https://github.com/benedekrozemberczki/SimGNN)
- [s-gwl NeurIPS'20](https://github.com/HongtengXu/s-gwl)