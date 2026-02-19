#!/bin/bash
set -ex

NUM_GPUS=${1:-4}
OUTPUT_DIR=${2:?  "Usage: $0 <output_dir>"}

python -m data.datasets --output_dir data/parquet

export LISTENER_MODELS="mistralai/Ministral-3-14B-Reasoning,microsoft/Phi-4-reasoning-plus,Qwen/Qwen3-14B"
export LISTENER_TEMPS="0.9,0.9,1.1"
export REWARD_TYPE="match_only"
export TRUNCATION_PCTS="0.25,0.50,0.75"
export NCCL_DEBUG=INFO
export VLLM_TORCH_COMPILE_LEVEL=0

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=data/parquet/train.parquet \
    data.val_files=data/parquet/val_bbh.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    custom_reward_function.path=training/reward_function.py \
    custom_reward_function.name=compute_cross_execution_reward \
    actor_rollout_ref.model.path=Qwen/Qwen3-14B \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model']" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    trainer.total_epochs=3 \
    trainer.project_name=remul_faithful \
    trainer.experiment_name=cross_execution_training \
    trainer.logger="['console','wandb']" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    trainer.val_before_train=True \
    trainer.default_local_dir=${OUTPUT_DIR}