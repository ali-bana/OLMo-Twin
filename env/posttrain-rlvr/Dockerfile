# Image for the two RLVR stages: the post-training image
# (env/posttrain/Dockerfile: open-instruct d19de7d2, torch 2.6.0+cu124)
# unchanged, plus the vLLM that the pinned requirements.txt names
# (vllm==0.8.2). Every other vLLM dependency is already in the base image from
# the same lock file. The pin xgrammar==0.1.16 no longer exists on PyPI; 0.1.17
# is the nearest release. xgrammar only serves guided decoding, which the GRPO
# script never uses. --no-deps keeps everything already in the base image
# (torch above all) unchanged.
#
# The InfiniBand user-space libraries let NCCL use an InfiniBand fabric between
# the two nodes of a run instead of falling back to TCP sockets. They matter
# only for multi-node NCCL over InfiniBand; elsewhere they are unused.
#
# This Dockerfile is a conversion of the Apptainer recipe from which the image
# used for our runs was built; commands and version pins are the same.
#
# Build the post-training image first, then, from the repository root:
#   docker build -f env/posttrain-rlvr/Dockerfile -t olmo-twin-posttrain-rlvr .
# (--build-arg BASE=<image> selects another tag of the post-training image).
ARG BASE=olmo-twin-posttrain
FROM ${BASE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libibverbs1 ibverbs-providers librdmacm1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --no-deps vllm==0.8.2 xgrammar==0.1.17
RUN python3 -c "import vllm, xgrammar, importlib.metadata as m; print('vllm', vllm.__version__, 'xgrammar', m.version('xgrammar'))"

ENV NLTK_DATA=/opt/nltk_data \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
