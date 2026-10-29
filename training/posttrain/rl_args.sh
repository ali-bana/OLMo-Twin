#!/usr/bin/env bash
# Single source of truth for the RLVR invocation (both stages), sourced by
# cache_datasets.sh rlvr and run_rlvr.sh so the dataset-cache hash is identical.
# Requires: OLMO_TWIN_* (from python3 -m olmo_twin.paths --shell), ARM,
# STAGE (1|2), MODEL_DIR, OUT.
#
# Verbatim from AI2's ORIGINAL 1B commands in open-instruct docs/olmo2.md
# (0423_grpo_seed_1 for stage 1, 0427_grpo_seed_1 for stage 2), except:
#   - local paths for the model, the tokenizer and the dataset cache;
#   - --with_tracking dropped (no wandb), --push_to_hub False added (AI2
#     pushed to the Hub), --try_launch_beaker_eval_jobs_on_weka dropped
#     (Beaker-only, a no-op here);
#   - --exp_name names the arm and stage.
TOK="$OLMO_TWIN_ASSETS/hf_models/OLMo-2-1124-7B-DPO-tokenizer"
case "$STAGE" in
    1) RL_DATA=allenai/RLVR-GSM-MATH-IF-Mixed-Constraints
       tok_args=(--tokenizer_name "$TOK" --tokenizer_revision main --tokenizer_name_or_path "$TOK") ;;
    2) RL_DATA=allenai/RLVR-MATH
       tok_args=(--tokenizer_name_or_path "$TOK" --tokenizer_revision main) ;;
    *) echo "ERROR: STAGE must be 1 or 2" >&2; return 2 ;;
esac
rl_args=(
    open_instruct/grpo_vllm_thread_ray_gtrl.py
    --exp_name "olmo2_1b_rlvr${STAGE}_${ARM}"
    --beta 0.01
    --local_mini_batch_size 32
    --number_samples_per_prompt 16
    --local_rollout_batch_size 4
    --kl_estimator kl3
    --learning_rate 5e-7
    --dataset_mixer_list "$RL_DATA" 1.0
    --dataset_mixer_list_splits train
    --dataset_mixer_eval_list "$RL_DATA" 16
    --dataset_mixer_eval_list_splits train
    --dataset_local_cache_dir "$OLMO_TWIN_POSTTRAIN/dataset_cache"
    --max_token_length 2048
    --max_prompt_token_length 2048
    --response_length 2048
    --model_name_or_path "$MODEL_DIR"
    --model_revision main
    "${tok_args[@]}"
    --use_slow_tokenizer False
    --add_bos
    --non_stop_penalty
    --stop_token eos
    --temperature 1.0
    --ground_truths_key ground_truth
    --chat_template_name tulu
    --sft_messages_key messages
    --total_episodes 2000000
    --penalty_reward_value 0.0
    --deepspeed_stage 2
    --per_device_train_batch_size 1
    --local_rollout_forward_batch_size 2
    --actor_num_gpus_per_node 4 8
    --num_epochs 1
    --vllm_tensor_parallel_size 4
    --lr_scheduler_type constant
    --apply_verifiable_reward true
    --seed 1
    --num_evals 100
    --save_freq 200
    --reward_model_multiplier 0.0
    --no_try_launch_beaker_eval_jobs
    --gradient_checkpointing
    --push_to_hub False
    --output_dir "$OUT"
)
