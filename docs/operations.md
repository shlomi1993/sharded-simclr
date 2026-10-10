# Running and inspecting experiments

Run commands from the repository root with the same Python environment active throughout. Python 3.12, PyTorch 2.9.1, torchvision 0.24.1, and PrettyTable 3 are the declared baseline. Linux CPU is the CI target; macOS supports the Gloo path but dependency/runtime combinations may need attention.

The local launch commands use an explicit IPv4 rendezvous endpoint with an automatically chosen port (`127.0.0.1:0`) and `--local-addr=127.0.0.1`. This avoids a macOS standalone-launch IPv6 hostname-resolution failure. These commands launch one local machine.

## Launch controls

| Option | Default | Meaning |
|---|---|---|
| `--local-batch-size` | 32 | Source images per replica pair |
| `--dataset-size` | 1024 | Synthetic source images |
| `--num-steps` | 10 | Total steps, including profiled-run warmup |
| `--split-layer` | `layer2` | Split after `layer1`, `layer2`, or `layer3` |
| `--profile` | Off | Export CPU profiler traces |
| `--run-name` | None | Output subdirectory name |
| `--output-dir` | `output` | Artifact base directory |
| `--overlap` | Off | Experimental deferred-backward scheduling |

Use `python train.py --help` for optimizer and projection options. Launch an even number of ranks of at least four. The dataset must contain at least one global source batch. Use more than two steps when profiling; the first two are warmup.

`OMP_NUM_THREADS=1` is a useful starting point to avoid CPU oversubscription with four processes. Increase threads only after inspecting available cores and measuring the effect.

## Artifacts

| File | Meaning |
|---|---|
| `run_config.json` | Requested configuration, global batch, source images/s |
| `metrics.csv` | Measured per-rank step timing and odd-rank losses |
| `traces/rankN.json` | Chrome trace from each rank |
| `controller_log.json` | Successful sweep runs and selected configuration |
| `manual_sweep_summary.csv` | Summarized completed manual sweep runs |
| `diagnosis_summary.md` | Generated throughput decision and trace evidence |

Warmup losses appear in console output but not in the measured metrics CSV. CSV step IDs start at zero for the measured segment. Use unique run names/output directories for independent experiments; repeated names overwrite artifacts, and old outputs can confuse manual comparisons.

Open individual JSON files with Perfetto's **Open trace file** action. Raw traces can be large; only compact historical configurations and metrics are committed under `examples/`.

## Sweeps and demo

`python controller.py --batch-sizes 4 8 --split-layers layer1 layer2 --num-steps 5 --dataset-size 128` runs four sequential configurations. The default controller grid is larger and can be expensive. The controller's `--nproc` defaults to four.

`bash scripts/sweep.sh` sweeps local batches 4, 8, 16, 32, 64, and 128. A failed run is skipped. Start with a smaller controller grid on memory-constrained machines. `bash demo.sh --help` lists walkthrough options; the demo creates no background server and prints locations of traces to open manually.

## Troubleshooting

- **Memory pressure:** reduce local batch size; each process constructs a full split model and even ranks retain activation graphs for backward.
- **Invalid rank count:** use four, six, eight, or another even count of at least four.
- **Missing measured steps:** choose at least three total steps with `--profile`.
- **Import failures:** confirm `python -m pip` installs into the interpreter used by `python -m torch.distributed.run`.
- **Duplicate OpenMP runtime on macOS:** create a clean environment with consistent binary dependencies. The scripts do not set `KMP_DUPLICATE_LIB_OK`; mixing incompatible runtime libraries should be resolved in the environment.
- **No traces:** training must include `--profile` and complete successfully before analysis.
