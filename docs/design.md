# Design decisions

## Purpose

Expose the mechanics of a sharded contrastive training system: where activations and gradients cross rank boundaries, how replicas agree on updates, and how split points and batch sizes affect measured throughput. Synthetic images keep execution self-contained and make the workload suitable for repeatable systems experiments.

## Decisions and tradeoffs

| Decision | Reason | Tradeoff |
|---|---|---|
| Explicit `torch.distributed` protocol | Makes communication and stage ownership inspectable | More protocol code and failure handling than DDP |
| Two sequential ResNet18 stages | Allows controlled changes to activation size and compute balance | Blocking handoffs introduce idle time |
| Replicate each stage across pairs | Combines model partitioning with data parallel gradient averaging | Every parameter is reduced individually |
| Detached remote embeddings | Avoids autograd-aware collectives | Gradient differs from exact global contrastive loss |
| FakeData and fixed seeds | No external dataset dependency | No semantic representation evaluation |
| CPU and Gloo baseline | Runs without GPU infrastructure | No GPU/NCCL scaling evidence |
| Named profiler spans | Connects runtime events to code paths | Annotated coverage and profiler overhead affect interpretation |
| Throughput-first sweep selection | Gives the controller an observable primary objective | Single short runs are noisy |

Parameters are broadcast initially and gradients averaged per stage. BatchNorm buffers are not synchronized. Short runs additionally compare the first parameter across same-stage replicas; this is a diagnostic rather than exhaustive state verification.

## Boundaries and next steps

The implementation has fixed-shape pair transfers, no failure recovery, no checkpoints, and no multi-node operating runbook. The controller launches local jobs and skips unsuccessful launches. It is a parameter sweep, not an online adaptive load balancer.

Useful extensions are one-stage-per-process construction, explicit device placement, CUDA profiling, differentiable embedding gather, synchronized or alternative normalization, repeated benchmarks with machine metadata, checkpoint/resume, and a real-image linear evaluation. These are future work; none is represented as an implemented capability.
