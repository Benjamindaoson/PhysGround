# AMD 192GB Stage-2 Runbook

Target image:

    ubuntu22.04-rocm7.2.3-py312-torch2.12.0-1.40.1

Do not upgrade ROCm or replace the image's preinstalled PyTorch before the
environment fingerprint is saved.

## Version locks

ROCm 7.2.3 is prepared against:

- JAX 0.8.2
- jaxlib 0.8.2
- ROCm 7 PJRT/plugin 0.8.2
- MuJoCo 3.13.1
- mujoco-mjx 3.13.1

The JAX version follows AMD's ROCm 7.2.x compatibility line. The JAX and core
environments are deliberately isolated.

## First command after boot

    cd PhysGround
    git pull
    bash scripts/amd/preflight.sh

This writes a JSON hardware/software fingerprint under artifacts/env.

## Create environments

Core:

    bash scripts/amd/bootstrap_core.sh

JAX/ROCm:

    bash scripts/amd/bootstrap_jax_rocm.sh

The JAX bootstrap fails if ROCm is not 7.2.x or if JAX only sees CPU.

## One-command Stage-2 pilot

    bash scripts/amd/run_stage2.sh

Default workload:

- 500,000 NumPy reduced-model transitions
- full reference PushBench acceptance
- JAX ROCm smoke with 65,536 parallel reduced transitions
- 500,000 JAX reduced-model transitions
- 5,000 classic MuJoCo pilot transitions

The MuJoCo count is intentionally only a pilot. We will measure real throughput
on the 8-core machine before deciding whether the next corpus should be 10k,
50k, or 100k. This avoids wasting the 95-hour allocation.

## Resume and provenance

Generated datasets are NPZ shards. Every dataset directory includes a manifest
with record counts and SHA256 hashes. Re-run with resume enabled to reuse
completed shards while still validating their row counts.

## Expected output

    artifacts/stage2/
      preflight-system.json
      preflight-core.json
      preflight-jax.json
      jax-smoke.json
      reference-benchmark.json
      data/
        reduced-500k/
        jax-reduced-500k/
        mujoco-pilot/

Artifacts are gitignored. Verified summaries can later be promoted into
docs/results.

## Data boundary

Already available before the cloud machine starts:

- 19 versioned PushBench reference cases
- reduced synthetic generator
- JAX reduced generator
- classic MuJoCo generator
- benchmark and acceptance logic

Not available until the machine actually runs:

- large MuJoCo transition corpus
- MJX high-throughput rollout corpus
- SO-101 real transitions

Do not claim those datasets or results before retained manifests exist.

## Stop conditions

Stop expensive runs if any of these occur:

1. ROCm is not 7.2.x.
2. JAX backend reports CPU.
3. JAX device list is empty.
4. MuJoCo import/smoke fails.
5. reference benchmark gate fails.
6. generated transitions contain NaN or Inf.
