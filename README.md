![Fathom: Reference Workloads for Modern Deep Learning](/fathom.png?raw=true)

[![build status](https://travis-ci.org/rdadolf/fathom.svg?branch=master)](https://travis-ci.org/rdadolf/fathom)

This paper contains a full description of the workloads, performance characteristics, and the rationale behind the project:

> R. Adolf, S. Rama, B. Reagen, G.Y. Wei, D. Brooks. "Fathom: Reference Workloads for Modern Deep Learning Methods."
[(Arxiv)](http://arxiv.org/pdf/FIXME.pdf)
[(DOI)](http://dx.doi.org/10.1109/IISWC.2016.FIXME)

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

