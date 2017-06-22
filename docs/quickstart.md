# Installing Prerequisites

Fathom requires a fair number of other software packages to use. TensorFlow is the obvious dependency, but there are a number of other support libraries which are mostly used for data processing and ingest. Deep learning algorithms operate on real data, so many of them have to do a substantial amount of work to turn raw inputs into a form they can process efficiently.

## TensorFlow

 - Python 2.6+
 - TensorFlow 1.x+ (artifact for paper required [TensorFlow 0.8.0rc0](https://github.com/tensorflow/tensorflow/releases/tag/v0.8.0rc0))

For TensorFlow, you can either download a pre-built binary or build from source. The latter is more involved, but can allow more flexibility in configuration (i.e.- you can pass specific options to the underlying math libraries which can affect performance). 

To build from source, you'll also need Bazel, Google's build system. Instructions can be found in the TensorFlow documentation.

The TensorFlow API is rapidly changing, and so it is possible for Fathom to break in small ways on unintended versions of TensorFlow. These issues tend to be about package imports and renaming rather than fundamentally devastating differences, so feel free to submit pull requests if you fix them on your own.

## Supporting libraries

Fathom needs several other python as well, mostly for pre-processing inputs. For all of these, you have your choice of methods for installing them:

 - `apt-get`: (or your favorite Linux distribution's package manager) This is a quick route, but be careful of versioning. Sometimes distributions lag a fair ways behind in version numbers.
 - `pip`: preferred package installer for Python
 - `conda`: If you're using an Anaconda distribution of python, this is probably your best bet for numpy, scipy, and scikit-learn. You'll need to use `pip` for librosa and tqdm, though (as Continuum doesn't support these packages).

You'll want to install the following list of packages. (You may have several of them installed already, and you shouldn't need to re-install&mdash;Fathom doesn't use any fancy features).

 - numpy (most)
 - scipy (for scikit-learn)
 - scikit-learn ([MemNet](/models/#memnet), [Speech](/models/#speech), [Autoenc](/models/#autoenc))
 - six ([Seq2Seq](/models/#seq2seq))
 - librosa ([Speech](/models/#speech))
 <!-- - tqdm ([Speech](/models/#speech)) -->
 - h5py* ([Speech](/models/#speech))

*For h5py, you'll also need libhdf5, which is the C++ backend for interfacing with HDF5-formatted files. This is usually available as a Linux package, but [building from source](https://support.hdfgroup.org/downloads/index.html) is also fine. Any recent version should work. In Ubuntu, the package you're looking for is `libhdf5-dev`.

## Atari emulation

[DeepQ](/models/#deepq) requires a bit more support than the other models. This is largely because it is interacting directly with a running Atari emulator. Consequently, you'll need both the emulator itself and OpenCV to run it.

The [Arcade Learning Environment (ALE)](http://www.arcadelearningenvironment.org/) is a clean, two-way interface between machine learning models and an Atari 2600 emulator. Installation instructions can be found in the [ALE Manual](https://github.com/mgbellemare/Arcade-Learning-Environment/raw/master/doc/manual/manual.pdf), but boil down to two steps: building the ALE C++ backend, and installing the python wrapper.


[OpenCV](http://opencv.org/) is a collection of image processing and computational geometry functions designed to support computer vision. You'll need both a 2.x version of the backend library and also the python interface wrapper. Many Linux distributions have a package for both (Ubuntu's are `libopencv-dev` and `python-opencv`), but you can also [build from source ](http://docs.opencv.org/2.4.13/doc/tutorials/introduction/linux_install/linux_install.html) and then use `pip` to install the `opencv-python` wrapper.

# Alternative: Quickstart via Docker

If you don't need accurate performance numbers right away, we also provide a pre-built [Docker image](https://hub.docker.com/r/rdadolf/fathom/) to make it easy to get familiar with the Fathom workloads.

If you're not familiar with Docker, you can think of it as a lightweight virtualization layer, similar to a VM but at a higher level of abstraction. Installation instructions can be found on the [docker website](https://www.docker.com/). To run the Fathom image interactively, use this:

```sh
docker run -it rdadolf/fathom
```

The image will automatically be downloaded from the Docker hub, launched, and you'll be given a shell prompt with the environment all set up.

# Downloading Data

*<div style='color: red'>Documentation in progress</div>*

Fathom does not come with datasets suitable for training. This is a combination of size (realistic training sets are often massive) and licensing (an oft-repeated mantra is that good data is more valuable than a good model).

Regardless, the inputs Fathom is designed for are standard and widely-available:

- [ImageNet](http://www.image-net.org/download-images) - requires registration, but downloads are free for non-commercial purposes.
- [WMT15](http://www.statmt.org/wmt15/translation-task.html) - freely available online, and automatically downloaded by Fathom
- [bAbI](https://research.facebook.com/research/babi/) - freely available online
- [MNIST](http://yann.lecun.com/exdb/mnist/) - freely available online, and automatically downloaded by Fathom.
- [TIMIT](https://catalog.ldc.upenn.edu/ldc93s1) - requires membership of the Linguistic Data Consortium (this is not free, but it is widely available in the research community).
- Atari "Breakout" ROM - Technically not freely available. In practice, it is [available online](https://www.google.com/search?q=atari+breakout+rom). You can also legally obtain this by dumping the memory of an Atari 2600 running a copy of Breakout you bought.

We eventually want to write synthetic datasets which allow users to run Fathom out of the box without requiring the above downloads.

# Running the Workloads

Fathom is a Python library with command-line shims. To use Fathom, you'll need to tell your Python installation where to find it. The easiest way is to adjust your `PYTHONPATH` environment variable:

```sh
$ git clone https://github.com/rdadolf/fathom.git
$ export PYTHONPATH=`pwd`/fathom
```

Once you've done that, you can either run the models directly (using the command-line shims):

```sh
$ cd fathom 
$ ./fathom/seq2seq/seq2seq.py
```

or you can use Fathom as a Python library directly in your scripts:

```python
from fathom import Seq2seq
model = Seq2seq()
model.setup()
model.run()
```

## ImageNet

The ImageNet workflow is finicky for training and the parameters and optimizations we have included do not reflect the state-of-the-art (e.g., batch normalization). Several users have reported issues with running the training flow out of the box, and we are currently working on resolving these issues.

If you do not want to download and set up ImageNet, then you can switch to using MNIST as provided in `fathom/imagenet/mnist.py`. Some of the models (e.g., VGG) may require modification because their convolutional kernels compress the smaller MNIST images too much.

## DeepQ

Note: [DeepQ](/models/#deepq) currently looks for its ROMs relative to Fathom's root directory. In practice, this will cause problems if you don't run in that directory. We are working on a more general configuration interface, but in the meantime, you should feel free to modify the `ROM_PATH` variable in `fathom/deepq/emulator.py`.

