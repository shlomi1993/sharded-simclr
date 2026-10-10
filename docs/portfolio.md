# CV and LinkedIn project copy

## Project title

Sharded Contrastive Training with PyTorch Distributed

## CV bullet

Implemented a two-stage ResNet18 contrastive training system using low-level PyTorch Distributed primitives, with manual activation/gradient exchange, stage-wise replica synchronization, per-rank profiling, and automated batch-size/shard-boundary tuning.

## LinkedIn description

Built a distributed contrastive training and profiling project that combines two-stage model sharding with replicated stages. Implemented process groups, activation and boundary-gradient transfers, embedding gathering, and gradient averaging directly with torch.distributed. Added per-rank Chrome traces, performance analysis, batch-size and split-layer sweeps, saved experiment artifacts, and integration tests. The self-contained CPU/Gloo workload uses synthetic images to study communication, waiting, and stage balance.

## GitHub metadata

Suggested repository name: `sharded-simclr-project`

Description: Manual ResNet18 sharding and contrastive training with PyTorch Distributed, per-rank profiling, and trace-driven batch-size and split-layer tuning.

Topics: `pytorch`, `distributed-training`, `distributed-systems`, `contrastive-learning`, `simclr`, `model-parallelism`, `profiling`, `python`.

Project link: https://github.com/shlomi1993/sharded-simclr-project

Use this URL for the CV and LinkedIn project link. Avoid adding GPU scaling or speedup claims until supported by repeatable new measurements.
