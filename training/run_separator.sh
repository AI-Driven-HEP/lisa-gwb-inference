#!/usr/bin/env bash
#
# One entrypoint for a Stage-1 training seed on a CPU cluster.
#
#   bash training/run_separator.sh --seed 1 --threads 2 --out runs/seed1
#
# Resumes automatically if --out already holds a checkpoint, so a job killed by
# a walltime limit is restarted by resubmitting exactly the same command. Keep
# the settings fixed when resuming so the same curriculum and schedule apply.
#
# The thread limits are exported BEFORE Python starts. That is the whole reason
# this script exists rather than the flags being passed to train_separator.py
# directly: OpenMP reads OMP_NUM_THREADS when its runtime initialises, which
# for PyTorch is at import, so a value set from inside Python is too late for
# the OpenMP pool even though torch.set_num_threads still works. Setting both
# here is the only way to be sure.
#
set -euo pipefail

SEED=1
THREADS=2
OUT=""
POOL="results/stage_c_pool"
EPOCHS=200
BATCH=64
LR=3.58e-4
STOP_AFTER=0
FLUSH_DENORMAL=1
EXTRA=()

usage() {
    cat <<'EOF'
Usage: bash training/run_separator.sh --out DIR [options]

  --seed N        random seed for this run (default 1). Use a different --out
                  for each seed; three seeds are expected to run concurrently.
  --threads N     CPU threads (default 2). NOT all cores: concurrent seeds
                  at one-thread-per-core each will contend and run slower than
                  conservative ones. Measured scaling on the development
                  machine was 1.35x from 1 to 2 threads and only 1.45x from 1
                  to 4, so wide is poor value. P6: how many seeds run at once
                  is set by MEMORY, not by cores -- peak RSS is 2.1-2.2 GB per
                  seed at batch 64, and three of them will not fit on a 6 GB
                  machine.
  --out DIR       output directory (required). Holds last.pt (resumable,
                  written every epoch), best.pt (best validation), the loss
                  history and this run's log. If last.pt is already there the
                  run RESUMES from it.
  --pool DIR      Stage-C pool directory (default results/stage_c_pool)
  --epochs N      total curriculum epochs (default 200). Keep it fixed across
                  a resume: it sets the cosine schedule.
  --batch N       minibatch size (default 64). Raised from 32 in P6: it
                  roughly doubles the per-batch effective sample size of the
                  pool's importance weights and costs nothing per epoch,
                  because the step count halves as the step cost doubles.
  --lr F          learning rate (default 3.58e-4). Set by the P6 scan at the
                  production width: it is the geometric middle of [2.34e-4,
                  5.48e-4], the range within which the scan's one seed per rate
                  cannot tell two rates apart. It is NOT the single best score
                  (5.48e-4 scored 0.0358 against 2.34e-4's 0.0397, a 10.9% gap
                  against a 17-23% epoch-to-epoch scatter). Replaces CAL1's
                  3e-4, which was inherited from width 32 and never measured
                  here.
  --flush-denormal / --no-flush-denormal
                  flush subnormal floats to zero (FTZ/DAZ). ON BY DEFAULT: it
                  is the production setting adopted in P6 section 6. Worth up
                  to 4.7x on a run whose weights have drifted subnormal, and
                  bit-identical in validation loss. Use --no-flush-denormal to
                  reproduce the P6 learning-rate scan, which ran without it.
                  The setting is recorded in the checkpoint and a resume that
                  disagrees with it is refused.
  --stop-after N  stop this process after N epochs and exit, leaving a
                  resumable checkpoint. For a queue with a walltime: pick N so
                  the job finishes inside it, then resubmit the same command.
  --              everything after this is passed through to train_separator.py

Exit codes: 0 finished or stopped cleanly, non-zero means the run failed and
the message above says why.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)       SEED="$2"; shift 2 ;;
        --threads)    THREADS="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --pool)       POOL="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --batch)      BATCH="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --stop-after) STOP_AFTER="$2"; shift 2 ;;
        --flush-denormal)    FLUSH_DENORMAL=1; shift ;;
        --no-flush-denormal) FLUSH_DENORMAL=0; shift ;;
        -h|--help)    usage; exit 0 ;;
        --)           shift; EXTRA=("$@"); break ;;
        *)            echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$OUT" ]]; then
    echo "ERROR: --out is required (the directory this seed writes to)." >&2
    usage >&2
    exit 2
fi

PY="${PYTHON:-python3}"

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: '$PY' not found on PATH. Activate the environment first, or" >&2
    echo "       set PYTHON=/path/to/python before running this script." >&2
    exit 2
fi
if [[ ! -d "$POOL" ]]; then
    echo "ERROR: pool directory '$POOL' does not exist." >&2
    echo "       Pass --pool <dir>; it must hold shards/, grid.npz and manifest.json." >&2
    exit 2
fi
if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [[ "$THREADS" -lt 1 ]]; then
    echo "ERROR: --threads must be a positive integer, got '$THREADS'." >&2
    exit 2
fi

mkdir -p "$OUT"

# Before Python starts -- see the header.
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TOKENIZERS_PARALLELISM=false

LOG="$OUT/train.log"
if [[ -f "$OUT/last.pt" ]]; then
    echo "RESUMING: $OUT/last.pt exists; continuing that run."
else
    echo "STARTING: no checkpoint in $OUT; this is a fresh run."
fi
echo "  seed=$SEED threads=$THREADS epochs=$EPOCHS batch=$BATCH lr=$LR" \
     "flush_denormal=$FLUSH_DENORMAL"
echo "  pool=$POOL out=$OUT log=$LOG"

ARGS=(
    --data pool
    --pool_dir "$POOL"
    --ckpt_dir "$OUT"
    --out "$OUT/best.pt"
    --seed "$SEED"
    --threads "$THREADS"
    --epochs "$EPOCHS"
    --batch "$BATCH"
    --lr "$LR"
)
if [[ "$STOP_AFTER" -gt 0 ]]; then
    ARGS+=(--stop_after "$STOP_AFTER")
fi
if [[ "$FLUSH_DENORMAL" -eq 1 ]]; then
    ARGS+=(--flush-denormal)
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    ARGS+=("${EXTRA[@]}")
fi

set +e
"$PY" -u -m lisa_gwb.train_separator "${ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -ne 0 ]]; then
    echo "" >&2
    echo "TRAINING FAILED (exit $rc). See $LOG for the error and training output." >&2
    exit $rc
fi
echo "done. checkpoints in $OUT (best.pt, last.pt), log at $LOG"
