# Image for supervised fine-tuning (SFT), DPO and the Hugging Face scorer
# (scoring/score_injection_v2_hf.py); also the base of the RLVR image
# (env/posttrain-rlvr/Dockerfile).
#
# It follows open-instruct's own Dockerfile at the pinned commit d19de7d2
# (2025-04-18, the week of AI2's OLMo 2 1B post-training runs) in what is
# installed: Python 3.10, torch 2.6.0+cu124, flash-attn 2.7.2.post1, then the
# uv-exported requirements.txt of that commit, then open-instruct itself in
# editable mode at /opt/open-instruct. The base image differs from
# open-instruct's. vllm (and its pin xgrammar==0.1.16, whose wheels are no
# longer installable) is needed only for RLVR generation; the SFT and DPO
# entry points import neither, so they are skipped here and added in the RLVR
# image. env/posttrain/pip-freeze.txt is the `pip freeze` of the image we used.
#
# This Dockerfile is a conversion of the Apptainer recipe from which the image
# used for our runs was built; base image, commands and version pins are the
# same. The recipe copied our patched open-instruct checkout into the image;
# here the checkout is cloned at the pinned commit and our patch applied.
#
# Build from the repository root (the build context must contain training/):
#   docker build -f env/posttrain/Dockerfile -t olmo-twin-posttrain .
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPEN_INSTRUCT_COMMIT=d19de7d27ac2af2cd1dcbf856dba1a2d077b37f4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip git curl ca-certificates && \
        rm -rf /var/lib/apt/lists/*

# open-instruct at the pinned commit, with our patch (training/posttrain/
# open-instruct.patch: keep intermediate checkpoints, take the run timestamp
# from OI_RUN_TIMESTAMP) and the DeepSpeed ZeRO-2 configuration used for DPO,
# which the pinned commit lacks.
RUN git clone https://github.com/allenai/open-instruct /opt/open-instruct && \
    git -C /opt/open-instruct checkout --detach "$OPEN_INSTRUCT_COMMIT" && \
    test "$(git -C /opt/open-instruct rev-parse HEAD)" = "$OPEN_INSTRUCT_COMMIT"
COPY training/posttrain/open-instruct.patch /tmp/open-instruct.patch
RUN git -C /opt/open-instruct apply /tmp/open-instruct.patch
COPY training/posttrain/ds_configs/stage2_no_offloading_accelerate.conf \
     /opt/open-instruct/configs/ds_configs/stage2_no_offloading_accelerate.conf

RUN python3 -m pip install --no-cache-dir --upgrade pip "setuptools<70.0.0" wheel packaging
RUN python3 -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
RUN python3 -m pip install --no-cache-dir psutil ninja einops
# MAX_JOBS=16 applies to all following build steps, as the recipe's export did.
ARG MAX_JOBS=16
RUN python3 -m pip install --no-cache-dir flash-attn==2.7.2.post1 --no-build-isolation
RUN grep -vE '^(vllm|xgrammar)==' /opt/open-instruct/requirements.txt > /tmp/requirements.filtered.txt && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.filtered.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124
RUN cd /opt/open-instruct && python3 -m pip install --no-cache-dir -e . --no-deps
# Sentence tokenizer data used by open-instruct; the run scripts rely on the
# image setting NLTK_DATA.
RUN mkdir -p /opt/nltk_data && python3 -m nltk.downloader -d /opt/nltk_data punkt punkt_tab || true

ENV NLTK_DATA=/opt/nltk_data \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8
