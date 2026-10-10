#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p output
SWEEP_OUTPUT="$(mktemp -d output/manual-sweep.XXXXXX)"
for BS in 4 8 16 32 64 128; do
    RUN_NAME="sweep_bs${BS}"
    if python -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --local-addr=127.0.0.1 --nproc_per_node=4 train.py \
        --local-batch-size "$BS" --num-steps 10 --dataset-size 2048 \
        --profile --output-dir "$SWEEP_OUTPUT" --run-name "$RUN_NAME"; then
        python analyze.py --run-dir "$SWEEP_OUTPUT/$RUN_NAME"
    else
        echo "Run failed for local batch $BS; skipping incomplete artifacts." >&2
    fi
done
python summarize_sweep.py --output-dir "$SWEEP_OUTPUT" --pattern 'sweep_bs*'
echo "Sweep artifacts: $SWEEP_OUTPUT"
