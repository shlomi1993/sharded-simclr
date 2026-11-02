#!/usr/bin/env bash
# Manual local batch-size sweep: profile one 4-process run per batch size, analyze each completed run, then write a CSV
# summary and diagnosis for the whole sweep.

set -euo pipefail
cd "$(dirname "$0")/.."

BATCH_SIZES=(4 8 16 32 64 128)
NPROC=4
NUM_STEPS=10
DATASET_SIZE=2048

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
LAUNCH=(
    python -m torch.distributed.run
    --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --local-addr=127.0.0.1
    --nproc_per_node="$NPROC"
)

# Unique directories prevent mixing this sweep with older runs.
mkdir -p output
SWEEP_OUTPUT="$(mktemp -d output/manual-sweep.XXXXXX)"

FAILED=()
for BS in "${BATCH_SIZES[@]}"; do
    RUN_NAME="sweep_bs${BS}"
    echo "Running $RUN_NAME with local batch $BS"
    if "${LAUNCH[@]}" train.py \
        --local-batch-size "$BS" --num-steps "$NUM_STEPS" --dataset-size "$DATASET_SIZE" \
        --profile --output-dir "$SWEEP_OUTPUT" --run-name "$RUN_NAME"; then
        python analyze.py --run-dir "$SWEEP_OUTPUT/$RUN_NAME"
    else
        echo "Run failed for local batch $BS; skipping incomplete artifacts." >&2
        FAILED+=("$BS")
    fi
done

python summarize_sweep.py --output-dir "$SWEEP_OUTPUT" --pattern 'sweep_bs*'
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed local batches: ${FAILED[*]}" >&2
fi
echo "Sweep artifacts: $SWEEP_OUTPUT"
