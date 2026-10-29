# Software environments

The experiment used two Python environments for the CPU steps and three
container images for the GPU steps (the RLVR image extends the post-training
image).

| Environment | Definition | Used for | Hardware |
|---|---|---|---|
| analysis | `env/analysis/requirements.txt` | `analysis/`, `figures/`, `training/checks/`, the unit tests | CPU |
| collection | `env/collection/requirements.txt` | `collection/`, `design/` | CPU; one GPU for three steps (below) |
| mid-training image | `env/midtrain/Dockerfile` | mid-training, the OLMo-native scorers, building the injected stream | 16 A100-80GB GPUs (2 nodes x 8) for training; one GPU per scoring task |
| post-training image | `env/posttrain/Dockerfile` | SFT, DPO, the Hugging Face scorer, checkpoint conversion, dataset caching | 8 GPUs for SFT and DPO; one GPU per scoring task |
| RLVR image | `env/posttrain-rlvr/Dockerfile` | the two RLVR stages | 16 GPUs (2 nodes x 8) per stage |

All images expect the data root to be mounted and `OLMO_TWIN_DATA` to point at
it, and this repository to be mounted or copied into the container (the
scripts locate their data through `olmo_twin/paths.py`, see `docs/data.md`).

## Analysis environment (CPU)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r env/analysis/requirements.txt
pip install -e .
```

The pins are the versions of the environment that produced the paper's
analysis outputs (Python 3.13.12; the pinned versions need Python 3.12 or
later). With the same Matplotlib version and fonts, the figures are
pixel-identical to the published ones (see the comments in the requirements
file).

Some steps need more than the pinned packages:

- `training/checks/check_instance_filter.py` executes OLMo's own
  instance-filter function from an OLMo checkout at commit `090253d`
  (`--olmo-repo` or `OLMO_REPO`) and needs `torch` (a CPU build suffices).
- `training/checks/compare_hf_branches.py` needs `torch` and network access
  to the Hugging Face Hub and olmo-checkpoints.org.
- `training/checks/check_lr_schedule.py` needs `wandb` (it reads the run's
  offline Weights & Biases log).
- `collection/rlvr_overlap_*.py` run in this environment plus `transformers`
  (only the tokenizer is loaded).
- `collection/tests/test_dolma_text_renderers.py` needs `beautifulsoup4` and
  `wikiextractor==3.0.8` from the collection environment.

The RLVR analysis (`analysis/rlvr/`) was run with the Python of the
post-training image (Python 3.10, numpy 1.26.4, pandas 2.2.3), not with this
environment.

## Collection environment

```bash
pip install -r env/collection/requirements.txt
docker run -d --rm --name grobid -p 8070:8070 --init lfoppiano/grobid:0.8.1
```

GROBID 0.8.1 converts the arXiv PDFs to TEI for the PDF form of the papers;
WikiExtractor must be version 3.0.8 (checked at import). The versions of the
other packages used during the original collection were not recorded.

Three steps of collection and design need one GPU: the embedding search
(`collection/olmo2_nonmember_audit.py embed`), the pre-treatment scoring used
for pairing (`scoring/score_pretreatment_logprobs.py`) and the model samples
of cohort X (`design/build_injection_manifest.py gen-samples`). The overlap
search over the training corpora is CPU-only but reads 8.3 TiB.

## GPU images

The Dockerfiles are conversions of the Apptainer recipes from which the images
used in our runs were built; base images, commands and version pins are the
same. They have not themselves been used to train. Build them from the
repository root, which must be the build context because the images copy our
patches from `training/`:

```bash
docker build -f env/midtrain/Dockerfile        -t olmo-twin-midtrain .
docker build -f env/posttrain/Dockerfile       -t olmo-twin-posttrain .
docker build -f env/posttrain-rlvr/Dockerfile  -t olmo-twin-posttrain-rlvr .   # after olmo-twin-posttrain
```

On systems that run Apptainer or Singularity instead of Docker, convert a
locally built image:

```bash
apptainer build olmo-twin-midtrain.sif docker-daemon://olmo-twin-midtrain:latest
```

### Mid-training image (`env/midtrain/Dockerfile`)

- Base: `nvcr.io/nvidia/pytorch:26.06-py3` (the digest of the base used is
  recorded in the Dockerfile; PyTorch 2.13.0a0).
- OLMo cloned at commit `090253dac6688f2532509daa7aa2eb5fae50e956` into
  `/opt/OLMo` and installed in editable mode with its `train` extra; the build
  checks the SHA-256 of AI2's stage-2 configuration.
- `numpy<2`, `pytest`, `zstandard==0.25.0`.
- `--build-arg APPLY_SHARED_FS_PATCH=1` applies
  `training/midtrain/olmo-shared-fs-checkpoint.patch`, needed only when
  several nodes write checkpoints to one shared file system. In our runs the
  patched checkout was mounted over `/opt/OLMo` instead; the launcher
  `training/midtrain/train.sh` takes the OLMo checkout from `OLMO_REPO`
  (`/opt/OLMo` in this image).
- Used for: `training/midtrain/` (stream building, training),
  `scoring/score_injection_v2.py` and `scoring/score_injection_v2_extended.py`.

No `pip freeze` of this image was recorded; OLMo's `train` dependencies were
installed unpinned on top of the base image.

### Post-training image (`env/posttrain/Dockerfile`)

- Base: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, Python 3.10.
- `torch==2.6.0` (CUDA 12.4 build), `flash-attn==2.7.2.post1`, then the
  `requirements.txt` of open-instruct at commit
  `d19de7d27ac2af2cd1dcbf856dba1a2d077b37f4` without `vllm` and `xgrammar`,
  then open-instruct itself (editable, `/opt/open-instruct`) with
  `training/posttrain/open-instruct.patch` applied and the DeepSpeed ZeRO-2
  configuration of the DPO stage added.
- `NLTK_DATA=/opt/nltk_data` with the `punkt` and `punkt_tab` data; the run
  scripts rely on the image setting it.
- `env/posttrain/pip-freeze.txt` is the `pip freeze` of the image we used
  (transformers 4.50.0, DeepSpeed 0.15.4, accelerate 1.2.1, ray 2.44.1,
  numpy 1.26.4, pandas 2.2.3).
- Used for: `training/posttrain/` (SFT, DPO, checkpoint conversion, dataset
  caching), `scoring/score_injection_v2_hf.py`, `analysis/rlvr/`.

### RLVR image (`env/posttrain-rlvr/Dockerfile`)

- The post-training image (`--build-arg BASE=...` selects its tag) plus
  `vllm==0.8.2` and `xgrammar==0.1.17`, installed without dependencies so that
  nothing in the base image changes (the pinned `xgrammar==0.1.16` is no
  longer available; xgrammar only serves guided decoding, which the GRPO
  script does not use).
- InfiniBand user-space libraries (`libibverbs1`, `ibverbs-providers`,
  `librdmacm1`), which matter only for multi-node NCCL over InfiniBand.
- Used for: `training/posttrain/run_rlvr.sh`.

## GPUs used in our runs

| Step | GPUs |
|---|---|
| Mid-training | 16 A100-SXM4-80GB (2 nodes x 8) |
| SFT, DPO (each arm) | 8 A100-SXM4-80GB |
| Each RLVR stage (each arm) | 16 A100-SXM4 (2 nodes x 8; 12 for training, 4 for sampling) |
| Scoring (each model or checkpoint) | 1 A100 |
