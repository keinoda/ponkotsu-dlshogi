FROM nvcr.io/nvidia/tensorrt:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN ln -s -f /bin/true /usr/bin/chfn
RUN echo "resolvconf resolvconf/linkify-resolvconf boolean false" | debconf-set-selections

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    pkg-config \
    software-properties-common \
    curl fish \
    build-essential libopenblas-dev \
    clang lld clangd \
    libjemalloc-dev libjemalloc2  && \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    ln -s /usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2 /usr/local/lib/libjemalloc.so && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN git clone https://github.com/sunfish-shogi/shogihome.git && \
    cd shogihome && \
    git checkout v1.21.5 && \
    npm install && \
    npm run usi-csa-bridge:install && \
    cd /tmp && \
    rm -rf shogihome

WORKDIR /root

RUN mkdir book && mkdir models

RUN git clone https://github.com/yaneurao/YaneuraOu

ENV LD_PRELOAD=/usr/local/lib/libjemalloc.so
