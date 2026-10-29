# Image for mid-training (the OLMo trainer), the OLMo-native scorers
# (scoring/score_injection_v2.py, scoring/score_injection_v2_extended.py) and
# the stream builder (training/midtrain/build_injection_stream_v2.py).
#
# This Dockerfile is a conversion of the Apptainer recipe from which the image
# used for our runs was built; base image, commands and version pins are the
# same. The runs themselves used that Apptainer image, not an image built from
# this file.
#
# Base image digest recorded when the image was built (27 Aug 2026):
#   nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1
# (PyTorch 2.13.0a0+8145d63 according to the image labels). To pin the base
# exactly, replace the tag in the FROM line by this digest.
#
# Build from the repository root (the build context must contain training/):
#   docker build -f env/midtrain/Dockerfile -t olmo-twin-midtrain .
# Optional: --build-arg APPLY_SHARED_FS_PATCH=1 applies
# training/midtrain/olmo-shared-fs-checkpoint.patch to /opt/OLMo. The patch is
# needed only when several nodes write checkpoints to one shared file system;
# in our runs the patched checkout (same commit plus this patch) was mounted
# over /opt/OLMo at run time instead.
FROM nvcr.io/nvidia/pytorch:26.06-py3

LABEL org.opencontainers.image.source="https://github.com/allenai/OLMo" \
      org.opencontainers.image.revision="090253dac6688f2532509daa7aa2eb5fae50e956"

ARG DEBIAN_FRONTEND=noninteractive
ARG APPLY_SHARED_FS_PATCH=0

RUN set -eu; \
    if ! command -v git >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y --no-install-recommends git ca-certificates; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# OLMo at the pinned commit; AI2's stage-2 configuration must have the
# recorded SHA-256.
RUN set -eu; \
    git clone --filter=blob:none https://github.com/allenai/OLMo.git /opt/OLMo; \
    git -C /opt/OLMo checkout --detach 090253dac6688f2532509daa7aa2eb5fae50e956; \
    test "$(git -C /opt/OLMo rev-parse HEAD)" = \
        "090253dac6688f2532509daa7aa2eb5fae50e956"; \
    printf '%s  %s\n' \
        095dabb7b20c9b6710abad3717a789c9d3109e070069b8643805bb407d2a2fd1 \
        /opt/OLMo/configs/official-0425/OLMo2-1B-stage2-seed42.yaml \
        | sha256sum --check --strict

# Optional shared-file-system checkpoint patch (see the header).
COPY training/midtrain/olmo-shared-fs-checkpoint.patch /tmp/olmo-shared-fs-checkpoint.patch
RUN set -eu; \
    if [ "$APPLY_SHARED_FS_PATCH" = "1" ]; then \
        git -C /opt/OLMo apply /tmp/olmo-shared-fs-checkpoint.patch; \
    fi

RUN set -eu; \
    python -m pip install --no-cache-dir "numpy<2" pytest "zstandard==0.25.0"; \
    python -m pip install --no-cache-dir -e "/opt/OLMo[train]"; \
    python -m pip check

ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TOKENIZERS_PARALLELISM=false

# Build-time test (the recipe's %test section).
RUN set -eu; \
    test "$(git -C /opt/OLMo rev-parse HEAD)" = \
        "090253dac6688f2532509daa7aa2eb5fae50e956"; \
    printf '%s  %s\n' \
        095dabb7b20c9b6710abad3717a789c9d3109e070069b8643805bb407d2a2fd1 \
        /opt/OLMo/configs/official-0425/OLMo2-1B-stage2-seed42.yaml \
        | sha256sum --check --strict; \
    python -c "import olmo, torch, yaml, zstandard; print(torch.__version__, zstandard.__version__)"

# Commands run from the OLMo checkout, as the recipe's %runscript did.
WORKDIR /opt/OLMo
