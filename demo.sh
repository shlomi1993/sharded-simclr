#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
NPROC=4
BASELINE_BS=8
FOLLOWUP_BS=4
NUM_STEPS=5
NO_WAIT=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) echo 'Usage: bash demo.sh [--no-wait] [--nproc N] [--baseline-bs N] [--followup-bs N] [--num-steps N]'; exit 0 ;;
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
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$NPROC" -ge 4 && $((NPROC % 2)) -eq 0 && "$NUM_STEPS" -gt 2 && "$BASELINE_BS" -gt 0 && "$FOLLOWUP_BS" -gt 0 ]] || {
    echo 'Need even nproc >= 4, num-steps > 2, and positive batch sizes.' >&2; exit 2;
}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# Unique directories prevent mixing this demonstration with older runs.
DEMO_OUTPUT="$(mktemp -d "${TMPDIR:-/tmp}/simclr-demo.XXXXXX")"
for RUN in baseline followup; do
    BS="$BASELINE_BS"
    [[ "$RUN" != followup ]] || BS="$FOLLOWUP_BS"
    echo "Running $RUN with local batch $BS"
    python -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --local-addr=127.0.0.1 --nproc_per_node="$NPROC" train.py \
        --local-batch-size "$BS" --dataset-size "$((BS * NPROC * 4))" \
        --num-steps "$NUM_STEPS" --profile --output-dir "$DEMO_OUTPUT" --run-name "$RUN"
    python analyze.py --run-dir "$DEMO_OUTPUT/$RUN"
    if [[ "$NO_WAIT" == false ]]; then
        read -r -p 'Press Enter to continue. '
    fi
done
python - "$DEMO_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
values = {name: json.loads((root/name/'run_config.json').read_text())['images_per_sec'] for name in ('baseline', 'followup')}
print(f"Baseline: {values['baseline']:.2f} source images/s")
print(f"Follow-up: {values['followup']:.2f} source images/s")
if values['baseline'] > 0:
    print(f"Follow-up / baseline: {values['followup']/values['baseline']:.3f}x (single-run observation)")
print(f'Artifacts: {root}')
print('Open traces/rank*.json from either run in https://ui.perfetto.dev/')
PY
