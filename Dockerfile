# Matterport3DSimulator
# Requires nvidia gpu with driver 396.37 or higher
FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

# Install cudnn
ENV CUDNN_VERSION=8.2.1.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.3 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.3 \
    && apt-mark hold libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    doxygen \
    curl \
    tmux \
    vim \
    git \
    libjsoncpp-dev \
    libepoxy-dev \
    libglm-dev \
    libosmesa6 \
    libosmesa6-dev \
    libglew-dev \
    libopencv-dev \
    python3-setuptools \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

#install conda and create vln environment
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
# ENV PATH=/opt/conda/bin:$PATH
# RUN conda create -n vln python=3.8
# SHELL ["conda", "run", "-n", "vln", "/bin/bash", "-c"]

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install  \
    backports.functools-lru-cache \
    certifi \
    cycler \
    decorator \
    matplotlib \
    networkx \
    numpy \
    pandas \
    pillow \
    pyparsing \
    python-dateutil \
    PyYAML==5.4 \
    six \
