FROM tensorflow/tensorflow:0.8.0rc0-devel
MAINTAINER Bob Adolf <rdadolf@gmail.com>

RUN apt-get update

### Software required for Fathom
RUN apt-get install -y python-scipy
RUN pip install scikit-learn
RUN pip install librosa
RUN apt-get install -y libhdf5-dev
RUN pip install h5py

# ALE
RUN apt-get install -y libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
RUN git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git /tmp/ALE
RUN mkdir /tmp/build && cd /tmp/build && \
    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF /tmp/ALE && make
RUN cd /tmp/ALE && pip install .
# OpenCV
RUN apt-get install -y libopencv-dev python-opencv

### Create a Fathom working environment
RUN mkdir /data
RUN useradd -ms /bin/bash fathom
RUN chown fathom /data
RUN chmod a+rwx /data
USER fathom
WORKDIR /home/fathom
RUN git clone https://github.com/rdadolf/fathom.git
ENV PYTHONPATH /home/fathom/fathom

