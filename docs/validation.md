# Validation record

What was checked, where it was checked and what wasn't covered.

## README walkthrough on macOS (2026-11-02)

**Machine:** Apple M2 Pro, 12 cores, 16 GB RAM, in a clean `.sharded-simclr-venv` created by `scripts/install.sh`, with no OpenMP workaround needed.

**Environment:** Python 3.12.4 · PyTorch 2.9.1 · torchvision 0.24.1 · PrettyTable 3.18.0 · pytest 8.4.2

Every command in the README was run as written.

| Check | Result |
|---|---|
| `scripts/install.sh`, both into a fresh directory and over an existing venv, plus `pip check` | ✅ Passed |
| Quick-start training and `analyze.py` | ✅ All documented files written; peak memory about 2.1 GB |
| Controller sweep (2 batch sizes × 2 splits) | ✅ 4 runs ranked, `controller_log.json` written, in 40 s |
| `scripts/sweep.sh` (local batches 4–128) | ✅ Completed in about 11 minutes, with summary and diagnosis written |
| `demo.sh` | ✅ Completed, in about 20 s |
| `pytest` | ✅ All 19 passed in 31 s |

Peak memory across the 4 processes, sampled every 0.5 s, was 2.0 GB at local batch 4, 2.8 GB at 8, 3.0 GB at 16, 3.6 GB at 32, 6.2 GB at 64 and 7.5 GB at 128.

## Local run on macOS (2026-10-10)

**Environment:** Python 3.12 · PyTorch 2.9.1 · torchvision 0.24.1 · PrettyTable 3.18.0 · pytest 8.4.2

| Check | Result |
|---|---|
| Editable install and `pip check` | ✅ Passed |
| Integration tests: synchronous training, deferred schedule, trace analysis, controller sweeps | ✅ All 19 passed in 51.70 s |
| Demo: two profiled runs on 4 processes, with a throughput comparison | ✅ Completed |
| Python syntax, shell syntax, local Markdown links, example file schemas, staged whitespace | ✅ Passed |

## Continuous integration

Linux CPU validation runs on [GitHub Actions](https://github.com/shlomi1993/sharded-simclr/actions/workflows/ci.yml). The live workflow status is the authority on CI results.

## Known caveats

> [!WARNING]
> **Duplicate OpenMP runtime.** The local Conda environment reported a duplicate OpenMP runtime, so local training used `KMP_DUPLICATE_LIB_OK=TRUE` as a workaround for that environment only. The repository's scripts and CI don't set this variable. Use a clean environment with a single OpenMP runtime. These results don't show that every macOS binary combination works without adjustment.

- **IPv6 stall fixed:** the original standalone launcher stalled on a macOS IPv6 hostname lookup. The documented launch commands, the controller, the demo and the test helpers now use an explicit IPv4 local rendezvous with an automatically chosen port.
- **No GPU testing:** nothing was run on a GPU.

---

**See also:** [Operations](operations.md) · [README](../README.md)
