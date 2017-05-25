# Functions are missing from `cv2`

You've probably installed the wrong python library. Unfortunately, the `cv2` package in PyPI is not related to OpenCV at all. It's a name-squatter who has managed to upload a useless, empty package. There are a couple of ways to install OpenCV:

1. Install from source by following the directions on the [OpenCV website](http://docs.opencv.org/2.4.13/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation).
2. Install via apt: `sudo apt-get install python-opencv`.
3. Install using Anaconda: `conda install opencv`.

# DeepQ can't find a ROM, but it's right there!

[DeepQ](/models/#deepq) currently looks for its ROMs relative to Fathom's root directory.
This is a bit hacky, and it will cause problems if you run anywhere else, regardless of whether you're using Fathom from the command-line or as a module.
We're planning on fixing this eventually, but in the meantime, there are two solutions:

1. Run from the Fathom root directory.

This should work:
```sh
$ git clone https://github.com/rdadolf/fathom.git
$ cd fathom
$ export PYTHONPATH=`pwd`
$ ./fathom/<model>/<model>.py
```

But this won't:
```sh
$ git clone https://github.com/rdadolf/fathom.git /tmp/fathom
$ export PYTHONPATH=/tmp/fathom
$ /tmp/fathom/fathom/<model>/<model>.py
```

2. Edit [DeepQ](/models/#deepq) to point to an absolute path.

The `ROM_PATH` variable in [emulator.py](https://github.com/rdadolf/fathom/blob/master/fathom/deepq/emulator.py) tells the model where to search for a ROM.
If you replace this variable with the absolute path to fathom, you should be able to run it anywhere.
For instance, this should work:

```sh
$ git clone https://github.com/rdadolf/fathom.git /tmp/fathom
```

```python
# in /tmp/fathom/fathom/deepq/emulator.py:
ROM_PATH='/tmp/fathom/fathom/deepq/roms/'
```

```sh
$ export PYTHONPATH=/tmp/fathom
$ python /tmp/fathom/fathom/deepq/deepq.py
```

# I found an issue with the Speech model!

Our implementation requires significant improvement, which we have not yet undertaken for lack of time.
