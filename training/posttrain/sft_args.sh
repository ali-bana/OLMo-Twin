#!/usr/bin/env bash
# Single source of truth for the SFT invocation. Sourced by both
# cache_datasets.sh sft (--cache_dataset_only) and run_sft.sh, so the
# dataset-cache config hash is identical in both.
# Requires: OLMO_TWIN_* (from python3 -m olmo_twin.paths --shell), ARM,
# MODEL_DIR, OUT set by the caller.
sft_args=(
    open_instruct/finetune.py
    --exp_name "olmo2_1b_sft_${ARM}"
    --model_name_or_path "$MODEL_DIR"
    --model_revision main
    # finetune.py at the pinned commit loads the tokenizer from
    # --model_name_or_path (--tokenizer_name is a deprecated field that it
    # ignores); both start models carry the released tokenizer files of
    # allenai/OLMo-2-0425-1B (convert_to_hf.sh).
    --tokenizer_name "$OLMO_TWIN_ASSETS/hf_models/OLMo-2-0425-1B"
    --tokenizer_revision main
    --use_slow_tokenizer False
    --add_bos
    --chat_template_name tulu
    --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0
    --dataset_local_cache_dir "$OLMO_TWIN_POSTTRAIN/dataset_cache"
    --max_seq_length 4096
    --per_device_train_batch_size 2
    --gradient_accumulation_steps 8
    --learning_rate 3e-5
    --lr_scheduler_type linear
    --warmup_ratio 0.03
    --weight_decay 0.0
    --num_train_epochs 2
    --seed 1
    --logging_steps 1
    # no --with_tracking / wandb: tracking is disabled; the losses are in the
    # training log.
    --checkpointing_steps 1000
    --keep_last_n_checkpoints -1
    --push_to_hub False
    --try_launch_beaker_eval_jobs False
    --output_dir "$OUT"
)
