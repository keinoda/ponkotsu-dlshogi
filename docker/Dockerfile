FROM nvcr.io/nvidia/tensorrt:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -s -f /bin/true /usr/bin/chfn
RUN echo "resolvconf resolvconf/linkify-resolvconf boolean false" | debconf-set-selections

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    pkg-config \
    software-properties-common \
    curl fish \
    build-essential libopenblas-dev libomp-20-dev \
    clang-20 lld-20 clangd-20 \
    libjemalloc-dev libjemalloc2  && \
    ln -s /usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2 /usr/local/lib/libjemalloc.so && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

COPY . /root/ponkotsu_wcsc33

RUN git clone https://github.com/yaneurao/YaneuraOu

ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so
