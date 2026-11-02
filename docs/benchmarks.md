# Benchmarks

What the numbers measure, where they can mislead, and how to report new results.

## Throughput

`images_per_sec` counts **global source images**, not augmented views:

```text
images_per_sec = measured steps × local batch × replica pairs / rank-0 wall time
```

| | Profiled runs | Non-profiled runs |
|---|---|---|
| **Warmup excluded** | First 2 steps | None |
| **Timer includes** | Profiler finalization and trace export | Training only |
| **Timer excludes** | Initialization, final metrics gathering | Initialization, final metrics gathering |

> [!NOTE]
> Profiled throughput is **instrumented** throughput, not pure steady-state training speed.

## Reading trace diagnostics

**Communication %.** This is the time in named communication spans divided by the time in all recognized compute, communication and optimizer spans. Sweep tools then average the percentage across ranks.

- **Waiting is inside communication.** Blocking receives stand in for waiting time, and they're already counted in communication time.
- **The tie-breaker counts waiting twice.** The controller's tie-breaker adds waiting on top of communication as a heuristic. The two numbers aren't separate slices of time.
- **Long spans can mean waiting.** A long communication span may come from waiting for the partner rank's compute, not from a larger transfer.

**Stage ratio.** The analyzer and the controller use different definitions:

| Tool | Stage 1 side | Stage 0 side |
|---|---|---|
| `analyze.py` | forward + loss | forward |
| Controller and manual summary | forward + loss | forward + backward |

> [!WARNING]
> Neither ratio is a complete, symmetric comparison of stage compute. Stage-1 backward has no `record_function` annotation yet, even though the analyzer recognizes that span name. Look at the rank timelines before deciding which stage is heavier.

The generated diagnosis suggests **possible** causes. On its own, it doesn't prove where the bottleneck is.

## Historical examples

The [preserved sweep](../examples/legacy_cpu_sweep/README.md) used a `layer2` split:

| Local batch | Source images/s |
|---:|---:|
| 4 | 5.76 |
| 8 | 5.65 |
| 16 | 5.46 |
| 32 | 5.21 |

> [!CAUTION]
> These are the original files, not new measurements for this repository. They have no hardware or software inventory, no repeat trials and no raw traces. Don't read them as evidence of a universal best batch size, GPU performance or a statistically significant speedup.

## Reporting new results

**Record the environment:**

- **System:** OS, CPU and RAM
- **Software:** Python and PyTorch versions
- **Run settings:** rank count, thread settings, split, batch size, seed, profiling on or off, and number of measured steps

**Use a sound method:**

- **Repeat trials:** run each in a fresh output directory and report the median and spread.
- **Separate goals:** use a profiled run to diagnose problems, and longer non-profiled repeated runs to measure throughput. Account for warmup explicitly.
- **Compare like with like:** only compare runs in the same mode, and don't mix up source images with views.
- **Report failures:** include out-of-memory and other failed configurations alongside the successful runs.

---

**See also:** [Operations](operations.md) · [Design decisions](design.md) · [README](../README.md)
