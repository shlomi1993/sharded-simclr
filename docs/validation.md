# Validation record

Validated locally on macOS on 2026-10-10 with Python 3.12, PyTorch 2.9.1, torchvision 0.24.1, PrettyTable 3.18.0, and pytest 8.4.2.

- Editable package installation and `pip check` passed.
- All 19 integration tests passed in 51.70 seconds, covering synchronous training, the experimental deferred schedule, trace analysis, and controller sweeps.
- The demo completed two four-process profiled runs and printed the throughput comparison.
- Python syntax, shell syntax, local Markdown links, example artifact schemas, and staged whitespace checks passed.

The existing local Conda binary environment reports a duplicate OpenMP runtime. Local training validation used `KMP_DUPLICATE_LIB_OK=TRUE` as an environment-specific workaround; this variable is not enabled by repository scripts or CI. Prefer a clean environment with one OpenMP runtime. These results do not establish that all macOS binary combinations work without adjustment.

The original standalone launcher stalled on a macOS IPv6 hostname lookup. The documented launch commands, controller, demo, and test helpers now select an explicit IPv4 local rendezvous and an ephemeral port.

Linux CPU validation runs through [GitHub Actions](https://github.com/shlomi1993/sharded-simclr/actions/workflows/ci.yml). The live workflow status is the source of truth for CI. No GPU run was performed.
