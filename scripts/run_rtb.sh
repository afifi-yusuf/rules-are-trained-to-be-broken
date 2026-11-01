#!/usr/bin/env bash
# GRPO + LoRA on the rtb environment, generation and training colocated on 1xH100
# (Databricks "Medium" Serverless GPU). Run from a SkyRL checkout; notebooks/20_train.py
# sets everything up and calls this.
#
#   MODEL_KEY=qwen35_9b       Qwen3.5-9B, FSDP, bf16             (default)
#   MODEL_KEY=qwen36_35b_a3b  Qwen3.6-35B-A3B MoE, Megatron, FP8 (blockwise, FP8 params)
#   MODEL_KEY=qwen35_0p8b     Qwen3.5-0.8B, FSDP, tiny batches   (A10 smoke test)
#
# Settings follow SkyRL's own examples:
#   examples/train/search/run_search.sh                          multi-turn tool env
#   examples/train/models/run_qwen3.5_0.8b.sh                    Qwen3.5 flags under FSDP
#   examples/train/megatron/run_megatron_dapo_qwen3.6_35b_a3b_lora.sh   35B-A3B LoRA (4x8 H100)
#   examples/train/fp8/run_fp8_hopper_blockwise_fp8param_qwen35_35b_a3b.sh   Hopper FP8
# NOT YET RUN ON A GPU. The 35B-A3B combination (Megatron + LoRA + FP8 params + one GPU)
# has no upstream example; the pilot decides whether it is viable.
#
# Required env: RTB_DIR (this repo), DATA_DIR (train/validation parquet), RUN_NAME.
# Optional env: SKYRL_PYTHON (python of a prebuilt SkyRL venv; the notebooks set it, else uv run),
#               MODEL_KEY, THINKING (true|false), CKPT_ROOT, LOGGER, MLFLOW_EXPERIMENT,
#               TRAIN_BATCH, N_SAMPLES, MAX_TURNS, MAX_GEN, EVAL_INTERVAL (0 = no evals), EVAL_BEFORE.
# Extra args are passed through as SkyRL config overrides.
set -euo pipefail
set -x

: "${RTB_DIR:?path to this repo}"
: "${DATA_DIR:?dir with train.parquet and validation.parquet from rtb.train.make_run_data}"
: "${RUN_NAME:?run name}"
: "${MODEL_KEY:=qwen35_9b}"
: "${THINKING:=true}"
: "${CKPT_ROOT:=/tmp/rtb_ckpts}"
: "${LOGGER:=$([ -n "${DATABRICKS_HOST:-}" ] && echo mlflow || echo console)}"
: "${MLFLOW_EXPERIMENT:=rules-are-trained-to-be-broken}"  # Databricks wants an absolute path: /Users/<you>/rtb
: "${TRAIN_BATCH:=32}"
: "${N_SAMPLES:=8}"
: "${MAX_TURNS:=20}"  # chosen from the 10-30 turn baseline sweep (README): rewarded rule-breaking peaks at 20
: "${MAX_GEN:=1024}"
: "${EVAL_INTERVAL:=20}"
# 35B-A3B: 248k-vocab logits for a 16k-token sequence are 8-16 GB at once, so cap at 12k.
if [ "$MODEL_KEY" = qwen36_35b_a3b ]; then : "${MAX_INPUT:=12288}"; else : "${MAX_INPUT:=16384}"; fi
# Kernel FIPS mode (Databricks): keep OpenCV's bundled OpenSSL from aborting the process.
if [ "$(cat /proc/sys/crypto/fips_enabled 2>/dev/null)" = "1" ]; then
  case ":${PYTHONPATH:-}:" in
    *":$RTB_DIR/scripts/fips_stubs:"*) ;;
    *) export PYTHONPATH="$RTB_DIR/scripts/fips_stubs${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
fi
# Give Ray workers time to start on slow first launches (default is short).
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"
# EVAL_BEFORE (true|false) controls the step-0 eval; it is off whenever evals are off.
if [ "$EVAL_INTERVAL" = "0" ]; then EVAL_BEFORE=false; EVAL_INTERVAL=100000; CKPT_EVERY=25
else : "${EVAL_BEFORE:=true}"; CKPT_EVERY=$EVAL_INTERVAL; fi

COMMON=(
  data.train_data="['${DATA_DIR}/train.parquet']"
  data.val_data="['${DATA_DIR}/validation.parquet']"
  trainer.algorithm.advantage_estimator="grpo"
  # No KL penalty: skips the reference-model pass (~20% of training compute, and a
  # second model copy under Megatron). Short LoRA runs stay close to the base model.
  trainer.algorithm.use_kl_loss=false
  trainer.algorithm.off_policy_correction.tis_ratio_type=token
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0
  trainer.policy.model.lora.rank=32
  trainer.policy.model.lora.alpha=32
  trainer.policy.model.lora.lora_sync_path="${CKPT_ROOT}/${RUN_NAME}/lora_sync"  # final adapter is kept from here
  trainer.policy.optimizer_config.max_grad_norm=0.5
  trainer.policy.language_model_only=true
  generator.inference_engine.language_model_only=true
  trainer.placement.colocate_all=true
  trainer.placement.policy_num_gpus_per_node=1
  generator.inference_engine.num_engines=1
  generator.inference_engine.tensor_parallel_size=1
  generator.inference_engine.backend=vllm
  generator.inference_engine.run_engines_locally=true
  generator.inference_engine.weight_sync_backend=nccl
  trainer.epochs=1
  trainer.update_epochs_per_batch=1
  trainer.train_batch_size=$TRAIN_BATCH
  trainer.policy_mini_batch_size=$TRAIN_BATCH
  trainer.max_prompt_length=1024
  generator.max_input_length=$MAX_INPUT  # 15 turns with thinking can outgrow 12k
  generator.sampling_params.max_generate_length=$MAX_GEN
  generator.sampling_params.temperature=1.0
  generator.sampling_params.top_p=1.0
  generator.sampling_params.stop='["</tool_call>", "</answer>"]'
  generator.eval_sampling_params.temperature=0.6
  generator.eval_sampling_params.stop='["</tool_call>", "</answer>"]'
  generator.eval_sampling_params.max_generate_length=$MAX_GEN
  generator.batched=false
  generator.use_conversation_multi_turn=true
  generator.append_eos_token_after_stop_str_in_multi_turn=true
  generator.n_samples_per_prompt=$N_SAMPLES
  generator.max_turns=$MAX_TURNS
  environment.env_class=rtb
  environment.skyrl_gym.max_env_workers=16
  trainer.eval_before_train=$EVAL_BEFORE
  trainer.eval_interval=$EVAL_INTERVAL
  trainer.eval_batch_size=500
  trainer.dump_eval_results=true
  trainer.ckpt_interval=$CKPT_EVERY
  trainer.max_ckpts_to_keep=2
  trainer.resume_mode=latest
  trainer.ckpt_path="${CKPT_ROOT}/${RUN_NAME}"
  trainer.export_path="${CKPT_ROOT}/${RUN_NAME}/exports"
  trainer.logger="$LOGGER"
  trainer.project_name="$MLFLOW_EXPERIMENT"
  trainer.run_name="$RUN_NAME"
)
if [ "$THINKING" = "false" ]; then
  COMMON+=(generator.chat_template_kwargs="{enable_thinking: false}")
fi

case "$MODEL_KEY" in
  qwen35_9b|qwen35_0p8b)
    EXTRA=fsdp
    if [ "$MODEL_KEY" = qwen35_9b ]; then MODEL=Qwen/Qwen3.5-9B; MICRO=2; else MODEL=Qwen/Qwen3.5-0.8B; MICRO=4; fi
    BACKEND=(
      trainer.policy.model.path="$MODEL"
      trainer.strategy=fsdp
      trainer.ref.language_model_only=true
      trainer.remove_microbatch_padding=false  # no sample packing for Qwen3.5 GDN layers under FSDP yet
      trainer.policy.optimizer_config.lr=3.0e-5
      trainer.micro_forward_batch_size_per_gpu=$MICRO
      trainer.micro_train_batch_size_per_gpu=$MICRO
      generator.inference_engine.gpu_memory_utilization=0.6
    )
    ;;
  qwen36_35b_a3b)
    EXTRA=megatron
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
    export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
    export VLLM_USE_DEEP_GEMM_E8M0=0
    # Leave FLA_TILELANG unset on Hopper (SkyRL notes the Triton GDN backward is broken there).
    BACKEND=(
      trainer.policy.model.path="Qwen/Qwen3.6-35B-A3B"
      trainer.strategy=megatron
      trainer.policy.optimizer_config.lr=1.0e-5
      trainer.policy.megatron_config.tensor_model_parallel_size=1
      trainer.policy.megatron_config.pipeline_model_parallel_size=1
      trainer.policy.megatron_config.context_parallel_size=1
      trainer.policy.megatron_config.expert_model_parallel_size=1
      trainer.policy.megatron_config.expert_tensor_parallel_size=1
      # Share the outer LoRA matrices across the 256 experts: ~1.9B -> ~0.5B adapter params
      # (GPU grads, CPU-offloaded Adam state, and vLLM-side LoRA all shrink accordingly).
      trainer.policy.megatron_config.lora_config.experts_shared_outer_loras=true
      trainer.policy.megatron_config.fp8=e4m3
      trainer.policy.megatron_config.fp8_recipe=blockwise
      trainer.policy.megatron_config.fp8_amax_compute_algo=most_recent
      trainer.policy.megatron_config.fp8_param="${FP8_PARAM:-true}"
      trainer.policy.megatron_config.ddp_config.fp8_param_gather="${FP8_PARAM:-true}"
      trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true
      trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0
      trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true
      trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true
      generator.inference_engine.fp8_weight_sync_mode=blockwise
      generator.inference_engine.distributed_executor_backend=mp
      generator.inference_engine.engine_init_kwargs='{"gdn_prefill_backend": "triton", "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}'
      generator.inference_engine.gpu_memory_utilization=0.7
      trainer.micro_forward_batch_size_per_gpu=1
      trainer.micro_train_batch_size_per_gpu=1
    )
    ;;
  *) echo "unknown MODEL_KEY=$MODEL_KEY" >&2; exit 2 ;;
esac

if [ -n "${SKYRL_PYTHON:-}" ]; then
  LAUNCH=("$SKYRL_PYTHON" -m rtb.train.main_rtb)  # prebuilt venv: Ray workers reuse this interpreter
else
  LAUNCH=(uv run --isolated --extra "$EXTRA" --with-editable "$RTB_DIR" -m rtb.train.main_rtb)
fi
"${LAUNCH[@]}" "${COMMON[@]}" "${BACKEND[@]}" "$@"
