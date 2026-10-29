#!/usr/bin/env bash
# Single source of truth for the DPO invocation, so the dataset-cache run
# (cache_datasets.sh dpo) and the training run (run_dpo.sh) hash identically.
# Requires: OLMO_TWIN_* (from python3 -m olmo_twin.paths --shell), ARM,
# MODEL_DIR, OUT set by the caller.
#
# Verbatim from open-instruct docs/olmo2.md "Preference Tuning (DPO) / 1B"
# (as on main in Sep 2026; the file is not present at the pinned commit),
# except: --num_epochs there is the newer name, the pinned commit takes
# --num_train_epochs; and we drop wandb, add a kept checkpoint ladder, and
# point at local paths.
dpo_args=(
    open_instruct/dpo_tune_cache.py
    --exp_name "olmo2_1b_dpo_${ARM}"
    --model_name_or_path "$MODEL_DIR"
    --model_revision main
    --tokenizer_name_or_path "$MODEL_DIR"
    --tokenizer_revision main
    --use_slow_tokenizer False
    --add_bos
    --chat_template_name tulu
    --dataset_mixer_list allenai/olmo-2-0425-1b-preference-mix 1.0
    --dataset_local_cache_dir "$OLMO_TWIN_POSTTRAIN/dataset_cache"
    --max_seq_length 2048
    --per_device_train_batch_size 8
    --gradient_accumulation_steps 2
    --learning_rate 2.5e-6
    --lr_scheduler_type linear
    --warmup_ratio 0.1
    --weight_decay 0.0
    --num_train_epochs 1
    --seed 111
    --gradient_checkpointing
    --use_lora False
    --dpo_loss_type dpo_norm
    --dpo_beta 5
    --logging_steps 1
    --checkpointing_steps 500
    --keep_last_n_checkpoints -1
    --push_to_hub False
    --try_launch_beaker_eval_jobs False
    --output_dir "$OUT"
)
