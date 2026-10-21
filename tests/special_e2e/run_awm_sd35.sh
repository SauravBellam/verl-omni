#!/usr/bin/env bash
# AWM diffusion e2e smoke test on a tiny SD3 checkpoint, V1 sync trainer, vllm_omni rollout.
#
# Flow: tiny-random SD3 -> jpeg_compressibility dummy data -> deterministic ODE rollout with
#       the `old` LoRA adapter (StableDiffusion3DGPOPipeline registered for awm) ->
#       AWMLoss.prepare_actor_batch -> NFT FSDP engine step (old / current / reference
#       velocities at shared forward timesteps, StableDiffusion3DGPO adapter) -> AWM loss ->
#       old-adapter EMA refresh.
#
# Requires: vllm-omni, diffusers. Builds the tiny SD3 checkpoint fully offline (random
# weights, no Hub access) at ~/models/tiny-random/stable-diffusion-3-tiny-random unless
# MODEL_PATH is overridden.
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/stable-diffusion-3-tiny-random}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_awm_sd3}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
ENGINE=vllm_omni
max_prompt_length=64

# Smoke: the portable native/SDPA pair; the GPU smoke group passes the same overrides.
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${MODEL_PATH}"

n_resp_per_prompt=2
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
train_batch_size=$((mini_bsz * n_resp_per_prompt))
synthetic_train_size=$((train_batch_size * TOTAL_TRAIN_STEPS))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${synthetic_train_size}" \
    --val_size 2

# SD3's CLIP tokenizer ships no chat template, and the diffusion agent loop applies
# one to every prompt; pass through the raw user content.
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.global_std=True \
    algorithm.train_timestep_range="[0,3]" \
    algorithm.train_timestep_count=2 \
    algorithm.old_policy_decay_schedule=linear_to_0_3 \
    algorithm.old_policy_update_interval=1 \
    actor_rollout_ref.model.algorithm=awm \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.custom_chat_template="\"${custom_chat_template}\"" \
    "actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: ${max_prompt_length}}}" \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    "actor_rollout_ref.model.target_modules=['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.awm_clip_range=1e-4 \
    actor_rollout_ref.actor.diffusion_loss.awm_kl_coef=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.max_prompt_embed_length=$((77 + max_prompt_length)) \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=awm-sd3-v1-sync-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    "$@"

echo "AWM SD3 diffusion e2e test passed (training completed successfully)."
