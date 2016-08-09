![Fathom: Reference Workloads for Modern Deep Learning](/fathom.png?raw=true)

[![build status](https://travis-ci.org/rdadolf/fathom.svg?branch=master)](https://travis-ci.org/rdadolf/fathom)

This paper contains a full description of the workloads, performance characteristics, and the rationale behind the project:

> R. Adolf, S. Rama, B. Reagen, G.Y. Wei, D. Brooks. "Fathom: Reference Workloads for Modern Deep Learning Methods."
[(Arxiv)](http://arxiv.org/pdf/FIXME.pdf)
[(DOI)](http://dx.doi.org/10.1109/IISWC.2016.FIXME)

## Workloads

Name     | Description
-------- | -----
Seq2Seq  | Direct language-to-language sentence translation. State-of-the-art accuracy with a simple, language-agnostic architecture.
MemNet   | Facebook's memory-oriented neural system. One of two novel architectures which explore a topology beyond feed-forward lattices of neurons.
Speech   | Baidu's speech recognition engine. Proved purely deep-learned networks can beat hand-tuned systems.
AutoEnc  | Variational autoencoder. An efficient, generative model for feature learning.
Residual | Image classifier from Microsoft Research Asia. Dramatically increased the practical depth of convolutional networks. ILSVRC 2015 winner.
VGG      | Image classifier demonstrating the power of small convolutional filters. ILSVRC 2014 winner.
AlexNet  | Image classifier. Watershed for deep learning by beating hand-tuned image systems at ILSVRC 2012.
DeepQ    | Atari-playing neural network from DeepMind. Achieves superhuman performance on majority of Atari2600 games, without any preconceptions.

## Running

#### Prerequisites:
- TensorFlow v0.8+
- ...

#### Running a single model:

Models can be run directly:
```
$ ./fathom/seq2seq/seq2seq.py
```

Or as a library:
```
$ python
>>> from fathom import seq2seq.TF_SEQ2SEQ as Seq2seq
>>> model = Seq2seq()
>>> model.setup()
>>> model.run()
```
