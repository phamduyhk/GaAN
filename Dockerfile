FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    vim \
    wget \
    unzip \
    zip \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel

RUN apt-get install -y build-essential git ninja-build ccache

RUN apt install -y python3.7
RUN mkdir -p /usr/phamduy
WORKDIR /usr/
RUN apt-get install wget
RUN apt-get install zip unzip
RUN wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN wget https://apache.org/dist/incubator/mxnet/1.5.0/apache-mxnet-src-1.5.0-incubating.tar.gz -P /usr
RUN tar -xvzf /usr/apache-mxnet-src-1.5.0-incubating.tar.gz
RUN mv /usr/apache-mxnet-src-1.5.0-incubating /usr/mxnet
COPY seg_ops_cuda/mxnet_op/seg_op.* /usr/mxnet/src/operator/contrib/

RUN version=3.14
RUN build=0
COPY install_mxnet_ubuntu_python.sh /usr/mxnet/
#RUN /usr/mxnet/install_mxnet_ubuntu_python.sh

#RUN set -exuo pipefail
RUN apt remove --purge --auto-remove cmake
RUN mkdir -p /usr/tmp
WORKDIR /usr/tmp
RUN wget https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz
RUN tar -xzvf cmake-3.14.0.tar.gz
RUN cd cmake-3.14.0/
RUN cmake-3.14.0/bootstrap
RUN make -j$(nproc)
RUN make install

RUN apt-get install -y --no-install-recommends libopenblas-dev
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --yes --no-install-recommends libopencv-dev
RUN apt-get install -y build-essential libatlas-base-dev
ADD config.mk /usr/mxnet/config.mk
WORKDIR /usr/mxnet
RUN make -j$(nproc)

# Add Python support

WORKDIR /usr/mxnet/python

RUN pip3 install -e .

ADD GraphSampler /usr/GraphSampler
RUN mkdir /usr/GraphSampler/build

WORKDIR /usr/GraphSampler/build
RUN cmake ..
RUN make
RUN mkdir /usr/mxgraph
WORKDIR /usr/GraphSampler
RUN python3 install.py

WORKDIR /usr/phamduy
