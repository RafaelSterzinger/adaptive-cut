#!/usr/bin/env bash
# Reproduce the paper's main results table (Table 1): 8 benchmarks x {1, 5}-shot.
#
#   GPUS="0 1 2 3" bash run_final.sh              # all 16 jobs, then the table
#   GPUS="0"       bash run_final.sh isic lung    # a subset of benchmarks
#   MAX_EPISODES=50 bash run_final.sh isic        # quick smoke test
#
# One job per (benchmark, k); each job runs every fold of the benchmark and
# writes results/k{k}/{benchmark}/results.json. Jobs are spread round-robin
# over $GPUS and a finished job is skipped on re-runs, so the script resumes.
# Seed 1, all folds, all episodes: the paper's protocol. Expect several GPU-days
# in total (LVIS-92i: 10 folds x 2300 episodes; 5-shot LVIS peaks at ~19 GB).
#
# Requires: $DINO_WEIGHTS (DINOv3 ViT-L/16 checkpoint), $DATA_ROOT
# (benchmark root, see README), and python with requirements.txt installed.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0}"
SHOTS="${SHOTS:-1 5}"
MAX_EPISODES="${MAX_EPISODES:-0}"
RESULTS="${RESULTS:-results}"
DATASETS="${*:-lvis coco isic suim isaid lung pascal_part paco_part}"

: "${DINO_WEIGHTS:?set DINO_WEIGHTS to the DINOv3 ViT-L/16 checkpoint}"
: "${DATA_ROOT:?set DATA_ROOT to the benchmark root}"

read -ra gpu_list <<< "$GPUS"
mkdir -p "$RESULTS/logs"
pids=()
i=0
for k in $SHOTS; do
  for ds in $DATASETS; do
    out="$RESULTS/k$k/$ds/results.json"
    if [[ -f "$out" ]]; then
      echo "skip $ds k=$k ($out exists)"
      continue
    fi
    gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    i=$((i + 1))
    log="$RESULTS/logs/${ds}_k${k}.log"
    echo "launch $ds k=$k on GPU $gpu -> $log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" eval.py --dataset "$ds" --shots "$k" \
        --seed 1 --max-episodes "$MAX_EPISODES" --output "$out" > "$log" 2>&1 &
    pids+=($!)
  done
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [[ $fail -ne 0 ]]; then
  echo "some jobs failed; see $RESULTS/logs/*.log. Re-run to resume." >&2
  exit 1
fi

# ---- aggregate: mean over folds per cell, mean over benchmarks for Avg ----
"$PYTHON" - "$RESULTS" <<'EOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
cols = [("lvis", "LVIS-92i"), ("coco", "COCO-20i"), ("isic", "ISIC"), ("suim", "SUIM"),
        ("isaid", "iSAID"), ("lung", "X-Ray"), ("pascal_part", "PASCAL"), ("paco_part", "PACO")]
# published baselines, transcribed from the paper (not reproduced here)
baselines = {
    1: {"FSSDINO": [33.8, 51.3, 53.0, 52.5, 47.2, 76.7, 50.3, 35.3],
        "INSID3":  [41.5, 58.3, 54.0, 53.4, 51.7, 78.6, 49.1, 38.5],
        "FROST":   [32.3, 48.1, 54.4, 58.1, 49.1, 84.8, 54.3, 36.9]},
    5: {"FSSDINO": [43.4, 60.6, 63.6, 62.8, 58.5, 78.7, 60.8, 45.8],
        "INSID3":  [47.5, 64.3, 63.9, 63.1, 60.1, 79.4, 58.4, 47.0],
        "FROST":   [47.5, 61.8, 66.7, 67.2, 61.0, 86.4, 68.0, 51.3]},
}

def fmt(v):
    return "--" if v is None else f"{v:.1f}"

lines = ["| Method | " + " | ".join(n for _, n in cols) + " | Avg |",
         "|---|" + "---:|" * (len(cols) + 1)]
for k in (1, 5):
    lines.append(f"| **{k}-shot** |" + " |" * (len(cols) + 1))
    for name, vals in baselines[k].items():
        lines.append(f"| {name} | " + " | ".join(fmt(v) for v in vals) + f" | {sum(vals)/len(vals):.1f} |")
    ours = []
    for key, _ in cols:
        f = root / f"k{k}" / key / "results.json"
        ours.append(json.load(f.open())["miou"] if f.exists() else None)
    avg = sum(ours) / len(ours) if all(v is not None for v in ours) else None
    lines.append("| **Ours** | " + " | ".join(fmt(v) for v in ours) + f" | {fmt(avg)} |")
table = "\n".join(lines)
(root / "table.md").write_text(table + "\n")
print(table)
print(f"\nwritten to {root/'table.md'}")
EOF
