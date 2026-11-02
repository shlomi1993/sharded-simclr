# Operations

How to run experiments, what files they produce, and how to fix common problems.

**Contents:** [Before you start](#before-you-start) · [Launch options](#launch-options) · [Output files](#output-files) · [Sweeps and demo](#sweeps-and-demo) · [Troubleshooting](#troubleshooting)

---

## Before you start

- **Working directory:** run every command from the repository root, in the same Python environment throughout.
- **Supported versions:** Python 3.12, PyTorch 2.9.1, torchvision 0.24.1 and PrettyTable 3.
- **Platforms:** Linux, macOS on Apple Silicon, and Windows through WSL. CI runs on Linux and macOS. Intel Macs are not supported because PyTorch 2.9 has no Intel macOS builds.
- **Single machine:** the launch commands in this repository run on one local machine.

> [!TIP]
> Start with `OMP_NUM_THREADS=1` so four processes don't oversubscribe the CPU. Raise it only after checking how many cores you have and measuring the effect.

<details>
<summary>Why the launch commands use <code>127.0.0.1:0</code></summary>

The commands set an explicit IPv4 rendezvous endpoint with an automatically chosen port (`127.0.0.1:0`) and `--local-addr=127.0.0.1`. This avoids a failure on macOS where the standalone launcher stalls while resolving the IPv6 hostname.

</details>

## Launch options

| Option | Default | Meaning |
|---|---|---|
| `--local-batch-size` | `32` | Source images per replica pair |
| `--dataset-size` | `1024` | Number of synthetic source images |
| `--num-steps` | `10` | Total steps, including warmup for profiled runs |
| `--split-layer` | `layer2` | Split after `layer1`, `layer2` or `layer3` |
| `--profile` | off | Export CPU profiler traces |
| `--run-name` | none | Name of the output subdirectory |
| `--output-dir` | `output` | Base directory for output files |
| `--overlap` | off | Experimental deferred-backward schedule |

Run `python train.py --help` for the optimizer and projection options.

**Requirements:**

- **Rank count:** an even number, at least 4.
- **Dataset size:** at least one global source batch.
- **Profiling:** more than 2 steps, because the first 2 are warmup.

## Output files

| File | Contents |
|---|---|
| `run_config.json` | Requested configuration, global batch size, source images/s |
| `metrics.csv` | Per-rank step timings and odd-rank losses for the measured steps |
| `traces/rankN.json` | Chrome trace from each rank |
| `controller_log.json` | Successful sweep runs and the selected configuration |
| `manual_sweep_summary.csv` | Summary of completed manual sweep runs |
| `diagnosis_summary.md` | Generated throughput decision and trace evidence |

- **Warmup losses** are printed to the console but left out of `metrics.csv`.
- **Step IDs** in the CSV start at 0 for the measured steps.
- **Viewing traces:** open a trace JSON in [Perfetto](https://ui.perfetto.dev/) with **Open trace file**. Raw traces can be large. Only compact configurations and metrics are committed under [`examples/`](../examples/).

> [!WARNING]
> Reusing a run name **overwrites** its files, and leftover outputs can confuse manual comparisons. Give each experiment its own run name or output directory.

## Sweeps and demo

**Controller sweep.** This runs four configurations one after another:

```bash
python controller.py --batch-sizes 4 8 --split-layers layer1 layer2 \
    --num-steps 5 --dataset-size 128
```

`--nproc` defaults to 4. The default grid is larger and can be slow, so on machines with limited memory, start with a small grid like the one above.

**Manual batch sweep.** This tries local batch sizes of 4, 8, 16, 32, 64 and 128, skipping any run that fails:

```bash
bash scripts/sweep.sh
```

**Walkthrough demo.** This prints the location of each trace for you to open. It doesn't start a background server.

```bash
bash demo.sh --help
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| **Out of memory** | Reduce `--local-batch-size`. Every process builds the full split model, and even ranks keep activation graphs for backward. |
| **Invalid rank count** | Use 4, 6, 8 or another even number of at least 4. |
| **No measured steps** | With `--profile`, use at least 3 total steps. |
| **Import errors** | Check that `python -m pip` installs into the same interpreter that `python -m torch.distributed.run` uses. |
| **Duplicate OpenMP runtime (macOS)** | Create a clean environment with consistent binary dependencies. The scripts don't set `KMP_DUPLICATE_LIB_OK`. |
| **No traces** | Train with `--profile` and let the run finish before analyzing it. |

---

**See also:** [Benchmarks](benchmarks.md) · [Architecture](architecture.md) · [README](../README.md)
