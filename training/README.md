# training/: mid-training with injected documents, checks, and post-training

This directory holds everything needed to train the treated model and its
post-trained versions:

* `midtrain/`: rebuild AI2's OLMo 2 1B stage-2 run (seed 42) with the
  injected documents in place, using AI2's published configuration and the
  unmodified OLMo trainer;
* `checks/`: checks run after training that the run read the intended data
  in the intended order and that its configuration matches AI2's;
* `posttrain/`: SFT, DPO and the two RLVR stages of both arms (treated model
  and control) with AI2's open-instruct recipes.

Nothing here contains machine-specific paths. Data locations come from the
data root (`$OLMO_TWIN_DATA`, default `<repo>/data`; see `olmo_twin/paths.py`
and `docs/data.md`). The shell scripts are plain bash: they run the same
`torchrun`, `accelerate launch` or `ray` commands that our runs executed
inside containers on a Slurm cluster, but they contain no scheduler or
container commands. Run them inside the environments of `env/` (or
equivalent ones), and start them on your nodes in whatever way your cluster
requires.

Two upstream checkouts are passed in by environment variable; we ship only
our patches to them:

| variable | checkout | pinned in |
|---|---|---|
| `OLMO_REPO` | [allenai/OLMo](https://github.com/allenai/OLMo) at `090253dac6688f2532509daa7aa2eb5fae50e956` | `midtrain/versions.env` |
| `OPEN_INSTRUCT_DIR` | [allenai/open-instruct](https://github.com/allenai/open-instruct) at `d19de7d27ac2af2cd1dcbf856dba1a2d077b37f4`, with `posttrain/open-instruct.patch` applied | `posttrain/versions.env` |

The GPU steps (training, post-training) need the hardware described below;
everything else runs on a CPU machine.

## midtrain/

| file | purpose |
|---|---|
| `versions.env` | pinned identity of the run: OLMo commit, path and SHA-256 of AI2's configuration, SHA-256 of the placement file, 23,852 updates, base image, stage-1 checkpoint URL |
| `download_stage1_checkpoint.sh` | downloads AI2's stage-1 endpoint (step 1907359: model, AdamW state, trainer state; 17.8 GB) and checks the pinned sizes and SHA-256 values |
| `prepare_stage2_assets.py` | maps the 576 HTTP data paths of AI2's stage-2 configuration to local files, downloads them restartably, writes the token-file manifest and a localized configuration (only `data.paths` changed); `verify` fails on any mismatch |
| `download_ai2_stage2_checkpoints.sh` | downloads AI2's stage-2 checkpoint ladders (steps 1000 to 23000 and 23852): `ingredient3` (seed 42, the control) or `ingredient1` (seed 42069, the retrained reference) |
| `build_injection_stream_v2.py` | the stream builder: `plan` (copies to training positions; run when the design was frozen, output in `bundle/stream_plan/`), `build`, `verify`, `make-config`, `make-eval-manifest`, `simulate` |
| `build_stream.sh` | checks the bundle, then `build`, `make-config` and `verify` |
| `train.sh` | preflight checks and the `torchrun` launch, once per node |
| `olmo-shared-fs-checkpoint.patch` | optional change to OLMo's checkpoint writer for several nodes saving to one network file system |
| `tests/` | `pytest training/midtrain/tests` (numpy, pyyaml, pytest) |

### 1. AI2's inputs

The commands below run from the repository root; the `eval` line defines
`OLMO_TWIN_DATA`, `OLMO_TWIN_ASSETS` and the other layout variables in the
shell (the scripts do this themselves).

```bash
export OLMO_TWIN_DATA=/path/to/data-root      # optional; default <repo>/data
eval "$(python3 -m olmo_twin.paths --shell)"
git clone https://github.com/allenai/OLMo /path/to/OLMo
git -C /path/to/OLMo checkout 090253dac6688f2532509daa7aa2eb5fae50e956
export OLMO_REPO=/path/to/OLMo
CFG=$OLMO_REPO/configs/official-0425/OLMo2-1B-stage2-seed42.yaml
A=$OLMO_TWIN_ASSETS

# stage-1 endpoint (the start of mid-training)
bash training/midtrain/download_stage1_checkpoint.sh

# the 576 stage-2 token files (206 GB, 51,491,007,838 tokens)
python training/midtrain/prepare_stage2_assets.py plan --official-config $CFG \
    --data-root $A/stage2_tokenized --manifest $A/configs/stage2_tokenized_manifest.json \
    --localized-config $A/configs/OLMo2-1B-stage2-seed42.local.yaml
python training/midtrain/prepare_stage2_assets.py download --manifest $A/configs/stage2_tokenized_manifest.json
python training/midtrain/prepare_stage2_assets.py verify --olmo-repo $OLMO_REPO --official-config $CFG \
    --localized-config $A/configs/OLMo2-1B-stage2-seed42.local.yaml \
    --manifest $A/configs/stage2_tokenized_manifest.json \
    --checkpoint-root $A/checkpoints/official_olmo_stage1_4001B \
    --checkpoint-alias $A/checkpoints/official_olmo_stage1_4001B-unsharded \
    --output $A/asset_verification.json

# AI2's seed-42 and seed-42069 checkpoint ladders (reference models, scoring)
bash training/midtrain/download_ai2_stage2_checkpoints.sh ingredient3
bash training/midtrain/download_ai2_stage2_checkpoints.sh ingredient1
```

The token manifest records each file's absolute local path; the stream
builder and the checks open the files through it. OLMo restores an unsharded
checkpoint only from a path that ends in `-unsharded`, so the stage-1
download also creates the alias `official_olmo_stage1_4001B-unsharded`, which
`train.sh` loads from.

### 2. Build and verify the injected stream

```bash
OLMO_REPO=$OLMO_REPO bash training/midtrain/build_stream.sh
```

This needs the frozen design in `bundle/`, including the token files of the
injected documents (`bundle/tokens/`, rebuilt by the design step and checked
against `bundle/SHA256SUMS`). OLMo's loader permutes the chunk indices once
with PCG64 seeded with 42 and update *u* reads positions [512(*u*-1), 512*u*)
of that order. The builder computes this order for AI2's 576 files and the
order the same loader produces for 12,212,224 chunks (the 23,852 updates),
and writes, into the slot that the second order reads at position *p*,
either AI2's chunk at position *p* of the first order or the injected chunk
that replaces it. The result is 64 files (200 GB) in `stream/data/`, read by
the unmodified loader. `make-config` writes AI2's configuration with only the
data paths, run name, output folder, checkpoint intervals and retention, and
tracking keys changed, and asserts that nothing else differs. `verify`
checks the SHA-256 of every shard, all 24,081 injected chunks against the
bundle token files, 20,000 sampled background chunks against AI2's files,
and the configuration diff.

### 3. Launch

```bash
# on each node; NODE_RANK 0 runs the preflight checks first
OLMO_REPO=$OLMO_REPO NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 MASTER_PORT=29500 \
    bash training/midtrain/train.sh
```

`train.sh` runs, from the OLMo checkout,

```
torchrun --nnodes=$NNODES --nproc-per-node=8 --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT scripts/train.py \
    $OLMO_TWIN_DATA/stream/config/olmo2-1b-injection-v2.yaml \
    --load_path=$OLMO_TWIN_DATA/assets/checkpoints/official_olmo_stage1_4001B-unsharded \
    --try_load_latest_save=true --save_overwrite=true --evaluators=[] \
    --run_name=olmo2-1b-injection-v2-olmo2-1b-injection-v2 \
    --save_folder=$OLMO_TWIN_DATA/run/checkpoints --save_interval_ephemeral=null
```

with `--wandb=null` added when `ENABLE_WANDB=0`. By default (`ENABLE_WANDB=1`),
Weights & Biases logs offline into the save folder, as in our run; the
learning-rate check below reads that log. The run name is the one our run
saved; another name only shows up as a difference in `compare_configs.py`.
Before launching, the preflight checks the OLMo commit, the SHA-256 of AI2's
configuration and of the placement file, the stage-1 checkpoint files
against their `SHA256SUMS`, and runs `build_injection_stream_v2.py verify`
on all injected chunks and 5,000 background positions (`SKIP_PREFLIGHT=1`
skips it; `train.sh --preflight-only` runs only the checks). Restarting the
same command resumes from the latest checkpoint in the save folder.
`OLMO_SHARED_FS=1` (default) is an upstream OLMo setting under which only
global rank 0 creates checkpoint directories, which is needed when all nodes
save to one file system.

Our run used 2 nodes with 8 A100-SXM4-80GB GPUs each and ran as five
launches, four of which resumed from the run's own checkpoints with
data-loader and random-number-generator state restored. As configured,
sharded checkpoints are written every 500 updates (the last three kept) and
unsharded ones every 1,000 updates (all kept).

### Differences from AI2's run

The run follows AI2's published configuration file. It differs from AI2's
actual seed-42 run, as recorded in the configuration stored in AI2's
checkpoints, in the following points (see Appendix D of the paper; the
checks below print them):

* **Data:** chosen positions of the stream hold injected documents; every
  other position holds AI2's chunk for that position.
* **Hardware and batch layout:** 16 GPUs with 32 sequences each (eight
  micro-batches of four) instead of 256 GPUs with two sequences each; the
  global batch is 512 sequences of 4,096 tokens in both.
* **Execution keys:** AI2's run used the non-fused loss, no flash attention,
  `torch.compile` and hybrid ZeRO-2 sharding with 32 replicas and
  block-and-size wrapping; our run used the fused loss, flash attention, no
  compilation and `SHARD_GRAD_OP` sharding without wrapping, as the published
  file specifies.
* **Trainer:** OLMo at the commit above; AI2's checkpoints do not record the
  commit of their run. The file sets the z-loss multiplier to 1e-5, but this
  trainer does not pass it to the loss function, which applies its default of
  1e-4.
* **Command-line overrides:** the load path (our local copy of the stage-1
  endpoint), no in-loop downstream evaluations, resumption from the latest
  checkpoint and no ephemeral checkpoints (our first two launches still
  wrote them); the run name, output folder, checkpoint intervals and tracking
  keys differ from AI2's.
* **Learning rate:** the trainer derives the decay horizon from the duration
  of 5e10 tokens, so the rate reaches zero at update 23,842 and stays zero
  for the last 11 of the 23,852 updates.

No optimizer hyperparameter, scheduler, data order, duration, loss weight or
precision setting differs.

### Shared file systems

The only change we made to the OLMo trainer is
`olmo-shared-fs-checkpoint.patch` (from the repository root: `git -C
"$OLMO_REPO" apply "$PWD/training/midtrain/olmo-shared-fs-checkpoint.patch"`). When several nodes save
checkpoints to one network file system, OLMo's polling for directories
created by rank 0 can stall on cached negative lookups; the patch creates the
checkpoint directories on every rank and replaces the polling by a probe. It
changes no loss, optimizer or data code and is needed only in that setting.

## checks/

These scripts read the finished run (`run/checkpoints/`), the built stream
and AI2's files, and print what they find; the numbers are those of Appendix
D. All take `--help`; inputs default to the data root.

| script | checks | our result |
|---|---|---|
| `check_data_order.py` | the saved `train_data/global_indices.npy` equals OLMo's PCG64(42) permutation; the per-rank data indices logged for updates 18001-23852 (the only ones kept) match it; 16 and 256 ranks read the same chunks per update | equal; 93,632 of 93,632 rows match |
| `check_injected_copies.py` | through the saved index file, every injected copy holds its document's tokens at its planned positions, followed by AI2's tokens in the last partial chunk (`--stride 1`, the default, checks all copies) | 12,414 of 12,414 copies (24,081 chunks; first injected chunk in update 240, last in update 23614) |
| `check_background_chunks.py` | random non-injected positions are byte-identical to AI2's chunks at those positions; sampled injected positions differ | 2,500 (`--seed 20261028`) + 1,200 (`--seed 777 --draw 1300 --keep 1200`) = 3,700 identical |
| `check_instance_filter.py` | re-applies OLMo's instance filter (a sequence of 1 to 13 tokens repeated at least 32 times masks the chunk's loss) to every injected chunk; needs `--olmo-repo` and torch | 43 masked chunks from 18 documents |
| `compare_configs.py` | key-by-key comparison of our saved configuration, AI2's published file and the configurations in AI2's seed-42 (steps 1000 and 23852) and seed-42069 checkpoints | the differences listed above |
| `compare_hf_branches.py` | Hugging Face branches of `allenai/OLMo-2-0425-1B` against the OLMo-native checkpoints, four tensors each, bitwise; `--main` compares the `main` revision with the stage-1 and stage-2 endpoints (needs network access) | all four tensors bitwise identical for the stage-1 branch, five ingredient-3 branches (steps 1000, 12000, 19000, 20000, 23852) and the ingredient-1 endpoint; `main` equals none of the endpoints |
| `check_lr_schedule.py` | the logged learning rate equals 7.4487e-5 (1 - *u*/23842) and zero from update 23842; reads one offline W&B log file (needs `wandb`) | no mismatch |

## posttrain/

| file | purpose |
|---|---|
| `versions.env`, `versions_rl.env` | open-instruct commit, patch, seeds, dataset and tokenizer revisions, start models |
| `sft_args.sh`, `dpo_args.sh`, `rl_args.sh` | the exact open-instruct arguments; sourced by both the caching and the training scripts |
| `open-instruct.patch` | our only changes to open-instruct (`finetune.py`, `dpo_tune_cache.py`): honour `--keep_last_n_checkpoints` (without the patch both scripts delete all intermediate checkpoints at the end of training) and take the run-directory timestamp from `OI_RUN_TIMESTAMP` instead of each rank's clock, so that all ranks write to one directory |
| `ds_configs/stage2_no_offloading_accelerate.conf` | the ZeRO-2 configuration of DPO, missing at the pinned commit; a copy of AI2's `configs/ds_configs/stage2_accelerate.conf` at open-instruct commit `fc7748e` |
| `convert_to_hf.sh` | OLMo-native checkpoint to Hugging Face format, with the released tokenizer files |
| `cache_datasets.sh` | tokenizes a dataset into open-instruct's cache without training (`sft`, `dpo`, `rlvr`) |
| `run_sft.sh`, `run_dpo.sh` | `accelerate launch` on one node with 8 GPUs |
| `run_rlvr.sh` | one RLVR stage on a two-node Ray cluster (`head` and `worker`) |
| `cpu_shims/triton.py` | makes `import triton` fail so that DeepSpeed does not crash on hosts without a GPU driver; used with `CPU_ONLY=1` for caching and conversion only |

### Setup

```bash
# from the repository root, after eval "$(python3 -m olmo_twin.paths --shell)"
git clone https://github.com/allenai/open-instruct /path/to/open-instruct
git -C /path/to/open-instruct checkout d19de7d27ac2af2cd1dcbf856dba1a2d077b37f4
git -C /path/to/open-instruct apply "$PWD/training/posttrain/open-instruct.patch"
export OPEN_INSTRUCT_DIR=/path/to/open-instruct

H=$OLMO_TWIN_ASSETS/hf_models
# released tokenizer files of the base model (copied into both start models)
hf download allenai/OLMo-2-0425-1B tokenizer.json tokenizer_config.json \
    special_tokens_map.json generation_config.json vocab.json merges.txt --local-dir $H/OLMo-2-0425-1B
# tokenizer that AI2's RLVR commands load (tokenizer files only)
hf download allenai/OLMo-2-1124-7B-DPO config.json tokenizer.json tokenizer_config.json \
    special_tokens_map.json --revision e34ea60adff2e575f4fe7569eaffd1b28509b6fd \
    --local-dir $H/OLMo-2-1124-7B-DPO-tokenizer
```

(`hf` is the command-line tool of `huggingface_hub`; older versions call it
`huggingface-cli`.) open-instruct downloads the datasets itself, by name and
at revision `main`, into `HF_HOME`, which the scripts set to
`$OLMO_TWIN_DATA/assets/hf_cache_posttrain`. Our runs resolved `main` to the
commits pinned in `versions.env` and `versions_rl.env`
(`allenai/tulu-3-sft-olmo-2-mixture-0225` @ `d91a0785`,
`allenai/olmo-2-0425-1b-preference-mix` @ `c6454f3d`,
`allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` @ `7dbd180f`,
`allenai/RLVR-MATH` @ `bd2a9355`); open-instruct records the resolved commit
as `dataset_commit_hash` in each cache directory's `config.json`, so compare
it after caching. A pinned revision can be fetched with, for example,
`hf download allenai/RLVR-MATH --repo-type dataset --revision bd2a93551b503a395fadd1a740d957559cfe6f3c`.

### Start models

```bash
OLMO_REPO=$OLMO_REPO bash training/posttrain/convert_to_hf.sh \
    $OLMO_TWIN_DATA/assets/checkpoints/official_stage2_ingredient3_seed42/step23852-unsharded \
    $OLMO_TWIN_DATA/assets/hf_models/OLMo-2-0425-1B-seed42-hf            # control
OLMO_REPO=$OLMO_REPO bash training/posttrain/convert_to_hf.sh \
    $OLMO_TWIN_DATA/run/checkpoints/step23852-unsharded \
    $OLMO_TWIN_DATA/assets/hf_models/olmo2-1b-injection-v2-final-hf      # treated
```

The control starts from AI2's seed-42 endpoint (the run that our injected run
replicates); Appendix D reports that its converted weight files are
byte-identical to the branch `stage2-ingredient3-step23852-tokens51B` of
`allenai/OLMo-2-0425-1B`.

### Training

```bash
for ARM in baseline injected; do
    ARM=$ARM bash training/posttrain/run_sft.sh
    ARM=$ARM bash training/posttrain/run_dpo.sh
done
# RLVR, per arm and stage, on two nodes:
#   node 1:  RAY_PORT=6379 bash training/posttrain/run_rlvr.sh head 1 baseline
#   node 2:  RAY_HEAD_IP=<node 1> RAY_PORT=6379 bash training/posttrain/run_rlvr.sh worker
# then the same with stage 2, and both stages for ARM=injected
```

Each arm runs mid-training endpoint -> SFT -> DPO -> RLVR1 -> RLVR2 with the
same recipe; only the start model, names and paths differ. SFT and DPO ran on
one node with 8 A100-SXM4-80GB GPUs (SFT: 2 epochs, 13,466 updates; DPO: 1
epoch, 2,905 updates; checkpoints every 1,000 and 500 updates). Each RLVR
stage ran on two nodes with 8 A100 GPUs each (2,604 steps, checkpoints every
200 steps). The GRPO script is not patched, so its output directories carry
the launch time. AI2's released `OLMo-2-0425-1B-Instruct` equals step 2000
of its second RLVR stage, not the final step.

If the GPU nodes have no internet access, run `cache_datasets.sh` on a
machine that has (sharing the data root) and export `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1` for the training runs, as we
did. The cache key includes the tokenizer path, so `cache_datasets.sh` uses
the same `ARM` and `MODEL_DIR` as training: for SFT the arm's start model
(open-instruct's `finetune.py` at the pinned commit takes the tokenizer from
`--model_name_or_path`; `--tokenizer_name` is a deprecated field that it
ignores), for DPO the arm's SFT model (so DPO caching follows SFT of that
arm), and for RLVR the 7B-DPO tokenizer directory (one cache for both arms).
`CPU_ONLY=1` puts `cpu_shims/` on `PYTHONPATH` for hosts without a GPU
driver.

### Recipe and the known deviation

The SFT and DPO arguments are the 1B commands of open-instruct's
`docs/olmo2.md` as on `main` in September 2026 (the file is not present at
the pinned commit), with local paths, kept checkpoints and no tracking; the
RLVR arguments are AI2's original 1B GRPO commands from the same file
(`0423_grpo_seed_1`, `0427_grpo_seed_1`), apart from names, tracking, upload
and Beaker options.

One deviation from AI2's logged SFT run is known: our SFT averages the loss
over the label tokens of each micro-batch (`reduce_loss=mean`, the default at
the pinned commit, because the command we followed passes no value), whereas
AI2's logged OLMo 2 1B SFT run and the version of `docs/olmo2.md` at commit
`89b4f62` sum it (`--reduce_loss sum`). Both arms use the mean, so the
comparison between the treated model and the control is not affected.
