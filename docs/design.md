# Design decisions

Why the project is built the way it is, what each choice costs, and where it stops.

## Goal

Make the mechanics of sharded contrastive training **visible**:

- **Boundaries:** where activations and gradients cross between ranks.
- **Agreement:** how replicas agree on updates.
- **Throughput:** how split points and batch sizes affect measured throughput.

Synthetic images keep everything self-contained and repeatable, so the project works as a systems experiment.

## Decisions and tradeoffs

| Decision | Why | Cost |
|---|---|---|
| Explicit `torch.distributed` protocol | Communication and stage ownership are easy to inspect | More protocol code and failure handling than DDP |
| Two sequential ResNet18 stages | Activation size and compute balance can be changed in a controlled way | Blocking handoffs create idle time |
| Each stage replicated across pairs | Combines model partitioning with data-parallel gradient averaging | Every parameter is reduced individually |
| Detached remote embeddings | No autograd-aware collectives needed | Gradient differs from the exact global contrastive loss |
| FakeData and fixed seeds | No external dataset dependency | No evaluation of representation quality |
| CPU + Gloo baseline | Runs without GPU infrastructure | No evidence about GPU/NCCL scaling |
| Named profiler spans | Runtime events map back to code paths | Annotation coverage and profiler overhead affect interpretation |
| Throughput-first sweep selection | The controller has one observable main objective | Single short runs are noisy |

## Replica consistency

- **Parameters** are broadcast once at the start.
- **Gradients** are averaged within each stage group.
- **BatchNorm buffers** are **not** synchronized.
- **Short runs** also compare the first parameter across replicas of the same stage. This is a diagnostic, not a full check of model state.

## Limits

- **Transfers:** pairs exchange fixed-shape tensors.
- **Reliability:** there's no failure recovery and no checkpoints.
- **Deployment:** there's no multi-node operating runbook.
- **Controller:** it launches local jobs and skips failed ones. It is a parameter sweep, not an online adaptive load balancer.

## Possible next steps

None of these are implemented:

- [ ] One stage per process instead of building both stages in every process
- [ ] Explicit device placement and CUDA profiling
- [ ] Differentiable embedding gather
- [ ] Synchronized or alternative normalization
- [ ] Repeated benchmarks that record machine metadata
- [ ] Checkpoint and resume
- [ ] Linear evaluation on real images

---

**See also:** [Architecture](architecture.md) · [Benchmarks](benchmarks.md) · [README](../README.md)
