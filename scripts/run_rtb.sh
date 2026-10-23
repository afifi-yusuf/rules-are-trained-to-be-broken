#!/usr/bin/env bash
# GRPO + LoRA on the rtb environment, generation and training colocated on 1xH100.
# Adapted from SkyRL's examples/train/search/run_search.sh and
# examples/train/lora/run_qwen2_5_0.5b_gsm8k_grpo_lora.sh.
#
# NOT YET RUN ON A GPU. Batch sizes, lengths and lr are first guesses for the pilot.
#
# Usage (from a SkyRL checkout, with this repo at $RTB_DIR):
#   uv run --project $RTB_DIR python -m rtb.train.make_run_data --frac 0.3 --seed 0 --out $DATA_DIR
#   RTB_DIR=... DATA_DIR=... RUN_NAME=f0.3_s0 bash $RTB_DIR/scripts/run_rtb.sh [hydra overrides...]
set -euo pipefail
set -x

: "${RTB_DIR:?path to this repo}"
: "${DATA_DIR:?dir with train.parquet and validation.parquet from rtb.train.make_run_data}"
: "${RUN_NAME:=rtb_pilot}"
: "${MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${CKPT_ROOT:=$HOME/ckpts}"
: "${LOGGER:=$([ -n "${WANDB_API_KEY:-}" ] && echo wandb || echo console)}"

uv run --isolated --extra fsdp --with-editable "$RTB_DIR" -m rtb.train.main_rtb \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.optimizer_config.lr=3.0e-5 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=1024 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</tool_call>", "</answer>"]' \
  generator.eval_sampling_params.temperature=0.6 \
  generator.eval_sampling_params.stop='["</tool_call>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.append_eos_token_after_stop_str_in_multi_turn=true \
  generator.n_samples_per_prompt=8 \
  generator.max_turns=10 \
  environment.env_class=rtb \
  environment.skyrl_gym.max_env_workers=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=20 \
  trainer.eval_batch_size=500 \
  trainer.ckpt_interval=20 \
  trainer.max_ckpts_to_keep=3 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="${CKPT_ROOT}/${RUN_NAME}" \
  trainer.export_path="${CKPT_ROOT}/${RUN_NAME}/exports" \
  trainer.logger="$LOGGER" \
  trainer.project_name="rules-are-trained-to-be-broken" \
  trainer.run_name="$RUN_NAME" \
  "$@"
