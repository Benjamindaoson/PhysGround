# PAI DSW GPU Runbook

PAI DSW may expose an AMD accelerator through the PyTorch HIP runtime while
omitting system-wide rocminfo, rocm-smi, hipconfig, and /opt/rocm tooling.

PhysGround therefore treats the preinstalled PyTorch accelerator as a valid
execution backend independently from the optional JAX/ROCm toolchain.

## Verify the framework runtime

Run:

    python - <<'PY'
    import torch
    print("torch", torch.__version__)
    print("hip", torch.version.hip)
    print("cuda", torch.version.cuda)
    print("available", torch.cuda.is_available())
    print("count", torch.cuda.device_count())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("name", repr(torch.cuda.get_device_name(0)))
        print("memory_gb", p.total_memory / 1024**3)
        print("arch_list", torch.cuda.get_arch_list())
        print("gcn_arch", getattr(p, "gcnArchName", None))
    PY

For AMD, PyTorch still exposes the torch.cuda API while torch.version.hip
identifies the HIP/ROCm build.

## Use the PyTorch-HIP path first

Do not block Stage 2 on JAX.

    bash scripts/cloud/bootstrap_torch_accel.sh

Then use screen for the long pilot:

    screen -S physground

Inside screen:

    cd /mnt/workspace/PhysGround
    bash scripts/cloud/run_stage2_torch.sh 2>&1 | tee /mnt/data/PhysGround/stage2-torch/run.log

Detach: Ctrl-A then D.

Reattach:

    screen -r physground

## Default pilot

- 1,048,576 transition GPU smoke
- 2,000,000 PyTorch-accelerated reduced transitions
- full PushBench acceptance
- 200 classic MuJoCo transitions for throughput measurement

MuJoCo remains a CPU high-fidelity verifier in this pilot. We measure the 200
transition throughput before scaling it.

## JAX

Only install JAX ROCm after confirming that a compatible ROCm userspace is
actually available. The upstream jax[rocm7-local] path expects a preinstalled
ROCm stack; framework-only HIP containers may not satisfy that requirement.

The PyTorch backend already gives PhysGround batched GPU dynamics and autograd
without modifying the managed platform runtime.
