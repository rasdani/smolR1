ARG CUDA_VERSION="12.4.1"
ARG CUDNN_VERSION=""
ARG UBUNTU_VERSION="22.04"
ARG MAX_JOBS=4

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    git-lfs \
    build-essential \
    ninja-build \
    libaio-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN git lfs install

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir -p /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Create Python environment
ARG PYTHON_VERSION="3.11"
RUN conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}" -y
ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Install PyTorch and other dependencies
ARG PYTORCH_VERSION="2.5.1"
ARG CUDA="124"
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Create directories for cache and data
RUN mkdir -p /root/.cache/huggingface
# RUN mkdir -p /workspace/open-r1/data

# Set default command
CMD ["/bin/bash"] 
