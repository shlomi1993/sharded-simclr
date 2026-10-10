# Historical CPU sweep artifacts

These files were copied unchanged from an earlier local run of the source project. They provide a small example of the artifact schema and an observed batch-size trend.

- `manual_sweep_summary.csv`: saved aggregate report for batches 4, 8, 16, and 32.
- `sweep_bs*/run_config.json`: original requested configuration and recorded throughput.
- `sweep_bs*/metrics.csv`: original measured rank timings and losses.

The configuration uses four ranks (inferred from two replica pairs in global batch and the four ranks in each CSV), `layer2`, synthetic 224×224 images, and profiling. Hardware metadata and raw traces are unavailable. Original path names are preserved as artifact provenance. The summary cannot be regenerated from this example alone; capture new traces with `--profile` and run `summarize_sweep.py` on those runs.

See [measurement limits](../../docs/benchmarks.md). Do not use these files as expected test performance.
