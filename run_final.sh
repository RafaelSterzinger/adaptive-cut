#!/usr/bin/env bash
# Reproduce our rows of the paper's main results table (Table 2): 9 benchmarks,
# 1- and 5-shot (PerMIS is 1-shot only: each episode has one support frame).
#
#   GPUS="0 1 2 3" bash run_final.sh              # all 17 jobs, then the table
#   GPUS="0"       bash run_final.sh isic lung    # a subset of benchmarks
#   MAX_EPISODES=50 bash run_final.sh isic        # quick smoke test
#
# One job per (benchmark, k); each job runs every fold of the benchmark and
# writes results/k{k}/{benchmark}/results.json. Jobs are spread round-robin
# over $GPUS and a finished job is skipped on re-runs, so the script resumes.
# Seed 1, all folds, all episodes: the main table's protocol (PerMIS episodes are
# fixed, so the seed does not affect them). Expect several GPU-days in total
# (LVIS-92i: 10 folds x 2300 episodes; 5-shot LVIS peaks at ~19 GB).
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
DATASETS="${*:-lvis coco isic suim isaid lung pascal_part paco_part permis}"

: "${DINO_WEIGHTS:?set DINO_WEIGHTS to the DINOv3 ViT-L/16 checkpoint}"
: "${DATA_ROOT:?set DATA_ROOT to the benchmark root}"

read -ra gpu_list <<< "$GPUS"
mkdir -p "$RESULTS/logs"
pids=()
i=0
for k in $SHOTS; do
  for ds in $DATASETS; do
    [[ $ds == permis && $k != 1 ]] && continue   # PerMIS has one support per episode
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

# ---- aggregate: mean over folds per cell, mean over the row's benchmarks for Avg ----
"$PYTHON" - "$RESULTS" <<'EOF'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
cols = [("lvis", "LVIS-92i"), ("coco", "COCO-20i"), ("isic", "ISIC"), ("suim", "SUIM"),
        ("isaid", "iSAID"), ("lung", "X-Ray"), ("pascal_part", "PASCAL"), ("paco_part", "PACO"),
        ("permis", "PerMIS")]

def fmt(v):
    return "--" if v is None else f"{v:.1f}"

def avg(vals, k):
    # PerMIS counts only at 1-shot; a row with a missing expected cell gets no Avg
    need = [v for (key, _), v in zip(cols, vals) if k == 1 or key != "permis"]
    return None if None in need else sum(need) / len(need)

lines = ["| Method | " + " | ".join(n for _, n in cols) + " | Avg |",
         "|---|" + "---:|" * (len(cols) + 1)]
for k in (1, 5):
    lines.append(f"| **{k}-shot** |" + " |" * (len(cols) + 1))
    ours = []
    for key, _ in cols:
        f = root / f"k{k}" / key / "results.json"
        ours.append(json.load(f.open())["miou"] if f.exists() else None)
    lines.append("| Ours | " + " | ".join(fmt(v) for v in ours) + f" | {fmt(avg(ours, k))} |")
table = "\n".join(lines)
(root / "table.md").write_text(table + "\n")
print(table)
print(f"\nwritten to {root/'table.md'}")
EOF
