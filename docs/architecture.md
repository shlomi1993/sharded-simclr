# Architecture

How processes are organized, what each training step does, and which module owns what.

**Contents:** [Process topology](#process-topology) · [Training step](#training-step) · [Contrastive loss](#contrastive-loss) · [Data](#data) · [Modules](#modules) · [Deferred schedule](#experimental-deferred-schedule)

---

## Process topology

Ranks are grouped into **pairs**. Each pair is one full model replica:

- **Even rank (`2k`)** runs stage 0, the first half of ResNet18.
- **Odd rank (`2k+1`)** runs stage 1, the second half plus the projection head and loss.

> [!IMPORTANT]
> The world size must be **even and at least 4**, so there are at least two replicas.

Every rank creates its communication groups in the same order: pair groups, then the stage-0 group, then the stage-1 group.

| Group | Members | Used for |
|---|---|---|
| **Pair** | One even + one odd rank | `send` / `recv` of step IDs, activations and gradients |
| **Stage 0** | All even ranks | Initial parameter broadcast, gradient `all_reduce` |
| **Stage 1** | All odd ranks | Initial broadcast, embedding `all_gather`, gradient `all_reduce` |
| **World** | All ranks | Metrics `gather`, final barrier |

**Model:** an untrained torchvision ResNet18 whose classifier is replaced by a projection head. Each process builds both stages, but only its assigned stage's parameters go into its optimizer.

**Runtime:** Gloo backend, CPU tensors.

## Training step

One synchronous step, seen from a single pair:

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

1. **Even rank:** builds two augmented views per image and runs the stage 0 forward pass.
2. **Handoff:** sends the activation, together with a step ID, to its odd partner.
3. **Odd rank:** runs stage 1, gathers embeddings from the other replicas, computes the loss and runs backward.
4. **Return:** sends the boundary gradient back, and the even rank runs the stage 0 backward pass.
5. **Sync:** each stage group averages its gradients, then every rank takes a local SGD step.

## Contrastive loss

- **Positive pairs** sit next to each other, so the target view for index `i` is `i ^ 1`.
- **Self-similarity** is masked out before cross-entropy.
- **Gradients:** each odd rank keeps its own gathered embeddings in the autograd graph and treats remote embeddings as constants.

> [!NOTE]
> This is a **local-gradient approximation**, not exact global SimCLR.

## Data

- **Slicing:** manual dataset slicing gives each replica pair distinct source indices.
- **Incomplete batches:** an incomplete global batch is dropped, and steps cycle through the usable data.
- **Augmentation:** seeds are rank-specific.

## Modules

| Module | Responsibility |
|---|---|
| [`train.py`](../train.py) | Seeds, stage construction, alignment, warmup, training, metrics gathering |
| [`src/groups.py`](../src/groups.py) | Process group initialization and rank topology |
| [`src/model.py`](../src/model.py) | Split boundaries, projection head, boundary tensor shapes |
| [`src/training_step.py`](../src/training_step.py) | Activation/gradient protocol, synchronization, SGD, deferred schedule |
| [`src/contrastive_loss.py`](../src/contrastive_loss.py) | Positive-pair targets, similarity masking, contrastive objective |
| [`src/augmentation.py`](../src/augmentation.py) | Crop, flip, jitter, grayscale, paired views |
| [`src/profiling.py`](../src/profiling.py) | CPU activity capture and Chrome trace export |
| [`src/metrics.py`](../src/metrics.py) | Per-rank CSV and configuration JSON |
| [`analyze.py`](../analyze.py) | Named-span aggregation and printed reports |
| [`controller.py`](../controller.py) | Sequential profiled launches and selection log |
| [`summarize_sweep.py`](../summarize_sweep.py) | CSV summary and Markdown diagnosis |

## Experimental deferred schedule

Enabled with `--overlap`. In this mode, the even rank:

1. Saves one pending activation.
2. Processes that activation's returned gradient on the **next** call, matching it by step ID.
3. Drains the final backward pass at the end of training.

> [!WARNING]
> Transfers still use **blocking** `send` / `recv`. This is not a multi-microbatch asynchronous pipeline. Check correctness and timings yourself before claiming a scheduling benefit.

---

**See also:** [Design decisions](design.md) · [Operations](operations.md) · [Benchmarks](benchmarks.md) · [README](../README.md)
