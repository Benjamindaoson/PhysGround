# AutoDL Cloud Runbook

PhysGround supports both AMD ROCm and NVIDIA CUDA cloud instances.

## First command after boot

Do not install packages first.

Run:

    cd ~/PhysGround
    git pull --ff-only
    bash scripts/cloud/preflight.sh

This records the actual Python interpreter, GPU vendor/model, driver stack,
PyTorch accelerator state, JAX state, ROCm state, and MuJoCo availability.

## Python discovery

AutoDL images may contain Python under Miniconda without exposing a
python3.12 executable on PATH. PhysGround now searches:

- python
- python3
- python3.12
- /root/miniconda3/bin/python
- /root/miniconda3/bin/python3
- common system Python paths

Python 3.11 or newer is required.

## NVIDIA path

If preflight shows an NVIDIA GPU, use:

    bash scripts/cloud/bootstrap_nvidia_jax.sh

The script selects the JAX CUDA wheel based on the NVIDIA driver:

- driver >= 580: CUDA 13 wheel
- older supported driver: CUDA 12 wheel

It uses a separate .venv-jax-cuda and does not replace the image PyTorch.

Long jobs should run inside screen:

    screen -S physground

Then:

    cd ~/PhysGround
    bash scripts/cloud/run_stage2_nvidia.sh 2>&1 | tee artifacts/stage2-nvidia/run.log

Detach with Ctrl-A then D. Reattach with:

    screen -r physground

## AMD path

If preflight shows ROCm 7.2.x, keep using:

    bash scripts/amd/bootstrap_jax_rocm.sh
    bash scripts/amd/run_stage2.sh

## Important

A screenshot or cloud product label is not treated as authoritative hardware
metadata. The runtime preflight is authoritative for experiment routing.
