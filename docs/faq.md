# Functions are missing from `cv2`

You've probably installed the wrong python library. Unfortunately, the `cv2` package in PyPI is not related to OpenCV at all. It's a name-squatter who has managed to upload a useless, empty package. There are a couple of ways to install OpenCV:

1. Install from source by following the directions on the [OpenCV website](http://docs.opencv.org/2.4.13/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation).
2. Install via apt: `sudo apt-get install python-opencv`.
3. Install using Anaconda: `conda install opencv`.
