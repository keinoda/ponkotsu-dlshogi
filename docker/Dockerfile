FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -s -f /bin/true /usr/bin/chfn
RUN echo "resolvconf resolvconf/linkify-resolvconf boolean false" | debconf-set-selections

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    sudo \
    ssh \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    software-properties-common \
    build-essential libopenblas-dev \
    clang lld clangd \
    tmux fish libjemalloc-dev libjemalloc2  && \
    ln -s /usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2 /usr/local/lib/libjemalloc.so && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
RUN pip3 install cython scipy mup

WORKDIR /tmp
RUN git clone https://github.com/jj1guj/DeepLearningShogi && \
    cd DeepLearningShogi && \
    git checkout use_mup && \
    pip install -v . && \
    rm -r ~/.cache/pip && \
    rm -rf /tmp/DeepLearningShogi

WORKDIR /tmp
RUN git clone https://github.com/jj1guj/YaneuraOu && \
    cd YaneuraOu && \
    git checkout tensorrt10_support && \
    cd source && \
    make -j $(nproc) tournament COMPILER=clang++ YANEURAOU_EDITION=YANEURAOU_ENGINE_DEEP_TENSOR_RT_UBUNTU ENGINE_NAME="FukauraOu" && \
    cp ./YaneuraOu-by-gcc /usr/local/bin && \
    rm -rf /tmp/YaneuraOu

WORKDIR /root
ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so