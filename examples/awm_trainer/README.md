# AWM Trainer

Last updated: 10/18/2026

This example post-trains `stabilityai/stable-diffusion-3.5-medium` with
AWM (Advantage Weighted Matching, [paper](https://arxiv.org/abs/2509.25050))
on the OCR task, using `vllm-omni` rollout and `Qwen/Qwen2.5-VL-3B-Instruct`
as the OCR reward model. It mirrors the
[DGPO example](../dgpo_trainer/README.md) and shares its SD3.5 adapters.

AWM is an online forward-process algorithm: rollouts stay on the deterministic
ODE sampler (noise level 0), the actor re-noises the final clean latents at a few
shared forward timesteps, and the per-sample flow-matching error `-log p` is
weighted by the GRPO group advantage through a PPO-style clipped ratio against
the `old` LoRA adapter. The `old` adapter is also the EMA copy of the policy
(`algorithm.old_policy_decay_schedule=linear_to_0_3`, the paper's
`min(0.001 n, 0.3)` rate) that the `awm_kl_coef` velocity-space KL pulls toward.

Loss knobs under `actor_rollout_ref.actor.diffusion_loss`:

- `awm_clip_range`: PPO clip range on `exp(old_dsm - dsm)`; `0` disables clipping.
- `awm_kl_coef`: coefficient of the velocity MSE toward the EMA (`old`) adapter.
- `ref_kl_coef`: coefficient of the velocity MSE toward the base model (adapters off).
- `adv_clip_max`: advantage clamp.

`algorithm.train_timestep_range` / `train_timestep_count` / `timestep_fraction`
pick the training timesteps exactly as for DGPO.

## Installation

Follow the [installation guide](../../docs/start/install.md), then install the
OCR reward dependency:

```bash
uv pip install -e ".[ocr]"
```

The script defaults to one node with 2 actor/rollout GPUs and 1 reward GPU
(`NUM_GPUS_ACTOR_ROLLOUT`, `NUM_GPUS_REWARD`).

## Prepare the dataset

Get the raw OCR prompts from the original Flow-GRPO repository
(https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr), place them under
`$WORKSPACE/data/ocr` (`WORKSPACE` defaults to `$HOME`), and convert them:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/sd3
```

## Run

```bash
bash examples/awm_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh
```

Unlike DGPO, AWM does not need whole rollout groups inside one micro batch; the
batch sizes in the script are kept equal to the DGPO recipe for a side-by-side.

## Tested configuration

GPU validation is pending (see the tracking issue
[#763](https://github.com/verl-project/verl-omni/issues/763)); the loss, batch
preparation and config validation are covered by
`tests/trainer/diffusion/test_awm_loss_on_cpu.py`.
