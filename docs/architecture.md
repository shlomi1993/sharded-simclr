# Architecture and training protocol

## Process topology

An even world size of at least four is required. Pair `k` contains global ranks `2k` and `2k+1`. All ranks create communication groups in the same order: pair groups, stage-0 group, stage-1 group. The runtime uses Gloo and CPU tensors.

| Group | Members | Operations |
|---|---|---|
| Pair | One even and one odd rank | `send` / `recv` of step IDs, activations, and gradients |
| Stage 0 | All even ranks | Initial parameter broadcast and gradient `all_reduce` |
| Stage 1 | All odd ranks | Initial broadcast, embedding `all_gather`, gradient `all_reduce` |
| World | All ranks | Metrics `gather` and final barrier |

The split uses an untrained torchvision ResNet18 with its classifier replaced by a projection head. Each process constructs both stages. Only the assigned stage's parameters enter its optimizer.

## Synchronous step

```mermaid
sequenceDiagram
    participant E as Even rank
    participant O as Paired odd rank
    participant S as Other odd ranks
    E->>E: Read local source slice and create interleaved positive pairs
    E->>E: Stage 0 forward
    E->>O: Step ID and detached activation
    O->>O: Set activation requires_grad and run stage 1
    O->>S: all_gather embeddings
    S->>O: Remote detached embeddings
    O->>O: Local contrastive loss and backward
    O->>E: Step ID and boundary gradient
    E->>E: Stage 0 backward
    Note over E,O: Average gradients within each stage group; local SGD update
```

Positive pairs occupy adjacent positions, so the target view index is `index ^ 1`. Each odd rank keeps its own gathered embeddings attached to autograd and treats remote embeddings as constants. The self-similarity entry is masked before cross-entropy. This is a local-gradient approximation, not exact global SimCLR.

Manual dataset slicing assigns distinct source indices to replica pairs. An incomplete global batch is dropped, and steps cycle through usable data. Augmentations use rank-specific seeds.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `train.py` | Seeds, stage construction, alignment, warmup, training, metrics gathering |
| `src/groups.py` | Process group initialization and rank topology |
| `src/model.py` | Split boundaries, projection head, boundary tensor shapes |
| `src/training_step.py` | Activation/gradient protocol, synchronization, SGD, deferred schedule |
| `src/contrastive_loss.py` | Positive-pair targets, similarity masking, contrastive objective |
| `src/augmentation.py` | Crop, flip, jitter, grayscale, paired views |
| `src/profiling.py` | CPU activity capture and Chrome trace export |
| `src/metrics.py` | Per-rank CSV and configuration JSON |
| `analyze.py` | Named-span aggregation and printed reports |
| `controller.py` | Sequential profiled launches and selection log |
| `summarize_sweep.py` | CSV summary and Markdown diagnosis |

## Experimental deferred schedule

With `--overlap`, even ranks save one pending activation, process its returned gradient on the next call, and drain the final backward at the end. Messages include a step ID to match returned gradients. Transfers still use blocking `send` and `recv`; this mode is not a multi-microbatch asynchronous pipeline. Validate correctness and timings independently before claiming a scheduling benefit.
