# Benchmark methodology and limits

## Throughput

`images_per_sec` counts global source images, not augmented views:

```text
measured steps × local batch × number of replica pairs / rank-0 measured wall time
```

For profiled runs, two warmup steps are excluded. The wall timer wraps the profiler context, including profiler finalization and trace export; therefore this is instrumented-run throughput, not pure steady-state training throughput. Initialization and final metrics gathering are excluded. Non-profiled runs include no warmup.

## Trace diagnostics

Communication percentages divide named communication span durations by all recognized annotated compute, communication, and optimizer spans, then the sweep tools average per-rank percentages. Blocking receives are a waiting proxy and are already included in communication time; the controller tie-breaker adds waiting again as a heuristic, not a disjoint time partition.

The analyzer's stage ratio compares stage-1 forward plus loss with stage-0 forward. The controller and manual summary compare stage-1 forward plus loss with stage-0 forward plus backward. Neither ratio is a full, symmetric stage compute ratio. Stage-1 backward currently has no `record_function` annotation despite the analyzer recognizing that span name. Consult rank timelines before diagnosing the heavier stage.

Larger communication spans can reflect waiting for a partner's compute, not just larger tensor transfers. The generated diagnosis suggests possible causes and does not establish a causal bottleneck by itself.

## Historical examples

The preserved sweep reports 5.76, 5.65, 5.46, and 5.21 source images/s for local batches 4, 8, 16, and 32 with a `layer2` split. These are original artifacts, not rerun measurements for this extraction. They lack hardware/software inventory, repeat trials, and retained raw traces. Do not infer a universal optimal batch, GPU performance, or statistically significant speedup from them.

## Reporting new results

Record OS, CPU, RAM, Python/PyTorch versions, rank count, thread settings, split, batch, seed, profile status, and measured steps. Run repeated trials in fresh output directories and report medians and spread. Use a profiled run for diagnosis and longer non-profiled repeated runs for throughput, allowing for warmup explicitly. Compare like-for-like modes and distinguish source images from views. Report out-of-memory or failed configurations alongside successful runs.
