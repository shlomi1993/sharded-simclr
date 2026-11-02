#!/usr/bin/env bash
# Baseline -> analysis -> follow-up -> comparison walkthrough: run two profiled trainings with different local batch
# sizes, analyze each, and compare their throughput.
#
# Usage: bash demo.sh [--no-wait] [--nproc N] [--baseline-bs N] [--followup-bs N] [--num-steps N]

set -euo pipefail
cd "$(dirname "$0")"

NPROC=4
BASELINE_BS=8
FOLLOWUP_BS=4
NUM_STEPS=5
NO_WAIT=false

usage() {
    echo 'Usage: bash demo.sh [--no-wait] [--nproc N] [--baseline-bs N] [--followup-bs N] [--num-steps N]'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --no-wait) NO_WAIT=true; shift ;;
        --nproc|--baseline-bs|--followup-bs|--num-steps)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "Expected integer after $1" >&2; exit 2; }
            case "$1" in
                --nproc) NPROC="$2" ;;
                --baseline-bs) BASELINE_BS="$2" ;;
                --followup-bs) FOLLOWUP_BS="$2" ;;
                --num-steps) NUM_STEPS="$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if (( NPROC < 4 || NPROC % 2 != 0 || NUM_STEPS <= 2 || BASELINE_BS <= 0 || FOLLOWUP_BS <= 0 )); then
    echo 'Need even nproc >= 4, num-steps > 2, and positive batch sizes.' >&2
    exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Unique directories prevent mixing this demonstration with older runs.
TMP_ROOT="${TMPDIR:-/tmp}"
DEMO_OUTPUT="$(mktemp -d "${TMP_ROOT%/}/simclr-demo.XXXXXX")"

run_config() {
    local run_name="$1" batch_size="$2"
    echo "Running $run_name with local batch $batch_size"
    python -m torch.distributed.run \
        --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --local-addr=127.0.0.1 --nproc_per_node="$NPROC" \
        train.py --local-batch-size "$batch_size" --dataset-size "$((batch_size * NPROC * 4))" \
        --num-steps "$NUM_STEPS" --profile --output-dir "$DEMO_OUTPUT" --run-name "$run_name"
    python analyze.py --run-dir "$DEMO_OUTPUT/$run_name"
    if [[ "$NO_WAIT" == false ]]; then
        read -r -p 'Press Enter to continue. '
    fi
}

run_config baseline "$BASELINE_BS"
run_config followup "$FOLLOWUP_BS"

python scripts/compare_runs.py "$DEMO_OUTPUT/baseline" "$DEMO_OUTPUT/followup"
echo "Artifacts: $DEMO_OUTPUT"
echo 'Open traces/rank*.json from either run in https://ui.perfetto.dev/'
