![Fathom: Reference Workloads for Modern Deep Learning](https://raw.githubusercontent.com/rdadolf/fathom/master/fathom.png)

[![build status](https://img.shields.io/badge/build-disabled-lightgray.svg)](https://travis-ci.org/rdadolf/fathom)
[![docs status](https://readthedocs.org/projects/fathom/badge/?version=latest)](http://fathom.readthedocs.io/en/latest/)

## Release: [`0.9-soft`](https://github.com/rdadolf/fathom/releases)

This release reflects the state of Fathom more or less as it was for the paper published in September 2016. We are currently developing a somewhat more user-friendly version, which you can track in the GitHub issue tracker. If you're eager to use Fathom as it is, please let us know.

## Workloads

This paper contains a description of the workloads, performance characteristics, and the rationale behind the project:

> R. Adolf, S. Rama, B. Reagen, G.Y. Wei, D. Brooks. "Fathom: Reference Workloads for Modern Deep Learning Methods."
[(Arxiv)](http://arxiv.org/abs/1608.06581)
<span style='color=gray'>(DOI)</span>

Name     | Description
-------- | -----
Seq2Seq  | Direct language-to-language sentence translation. State-of-the-art accuracy with a simple, language-agnostic architecture.
MemNet   | Facebook's memory-oriented neural system. One of two novel architectures which explore a topology beyond feed-forward lattices of neurons.
Speech   | Baidu's speech recognition engine. Proved purely deep-learned networks can beat hand-tuned systems.
Autoenc  | Variational autoencoder. An efficient, generative model for feature learning.
Residual | Image classifier from Microsoft Research Asia. Dramatically increased the practical depth of convolutional networks. ILSVRC 2015 winner.
VGG      | Image classifier demonstrating the power of small convolutional filters. ILSVRC 2014 winner.
AlexNet  | Image classifier. Watershed for deep learning by beating hand-tuned image systems at ILSVRC 2012.
DeepQ    | Atari-playing neural network from DeepMind. Achieves superhuman performance on majority of Atari2600 games, without any preconceptions.

## Getting Started

We've begun to put together some actual documentation, and it's already better than what we had:

 - [Fathom Quickstart Guide](http://fathom.readthedocs.io/en/latest/quickstart/).

Please bear with us as the docs get fleshed out.
