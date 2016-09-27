![Fathom: Reference Workloads for Modern Deep Learning](/fathom.png?raw=true)

[![build status](https://travis-ci.org/rdadolf/fathom.svg?branch=master)](https://travis-ci.org/rdadolf/fathom)

## Release: `0.9-soft`

This release reflects the state of Fathom more or less as it was for the paper published paper in September 2016. We are currently developing a somewhat more user-friendly version, which you can track in the GitHub issue tracker. If you're eager to use Fathom as it is, please let us know. We might be able to help you get started.

## Workloads

This paper contains a full description of the workloads, performance characteristics, and the rationale behind the project:

> R. Adolf, S. Rama, B. Reagen, G.Y. Wei, D. Brooks. "Fathom: Reference Workloads for Modern Deep Learning Methods."
[(Arxiv)](http://arxiv.org/pdf/FIXME.pdf)
[(DOI)](http://dx.doi.org/10.1109/IISWC.2016.FIXME)

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

## Data

Fathom does not come with datasets suitable for training. This is a combination of size (realistic training sets are often massive) and licensing (an oft-repeated mantra is that good data is more valuable than a good model).
Regardless, the inputs Fathom is designed for are standard and widely-available.

These links should take you to the original data owners:

- [ImageNet](http://www.image-net.org/download-images) - requires registration, but downloads are free for non-commercial purposes.
- [WMT15](http://www.statmt.org/europarl/)
- [bAbI](https://research.facebook.com/research/babi/)
- MNIST - automatically downloaded by Fathom.
- TIMIT - requires membership of the Linguistic Data Consortium (this is not free, but it is widely available in the research community).
- Atari "Breakout" ROM - available online

## Running

#### Prerequisites:
Fathom is tested with TensorFlow v0.8rc0, and due to API instability, there are issues with recent versions (Google has changed TF's layout several times). If you're willing to rename your functions and swap a couple import statements, recent versions of TF should work.

Many of the models require external python libraries (e.g., [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) for DeepQ). Most of these are available as `pip` packages.

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
