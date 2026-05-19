#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?ground-truth root is required}"
OUT="${2:?output dir is required}"
TESSDATA="${3:?tessdata dir is required}"
MODEL_NAME="${4:-lenta_rus_q075}"
MAX_ITERATIONS="${5:-400}"

rm -rf "$OUT"
mkdir -p "$OUT"

combine_tessdata -e "$TESSDATA/rus.traineddata" "$OUT/rus.lstm" > "$OUT/combine_extract.log" 2>&1

for split in train val; do
  : > "$OUT/${split}.list"
  : > "$OUT/${split}_lstmf_generation.log"
  ok=0
  fail=0
  while IFS= read -r img; do
    base="${img%.tif}"
    rm -f "$base.lstmf"
    if tesseract "$img" "$base" --tessdata-dir "$TESSDATA" -l rus --psm 7 lstm.train >> "$OUT/${split}_lstmf_generation.log" 2>&1; then
      if [[ -s "$base.lstmf" ]]; then
        echo "$base.lstmf" >> "$OUT/${split}.list"
        ok=$((ok + 1))
      else
        echo "missing_lstmf $img" >> "$OUT/${split}_lstmf_generation.log"
        fail=$((fail + 1))
      fi
    else
      echo "failed $img" >> "$OUT/${split}_lstmf_generation.log"
      fail=$((fail + 1))
    fi
  done < <(find "$ROOT/ground-truth/$split" -name "*.tif" | sort)
  echo "$split ok=$ok fail=$fail" | tee "$OUT/${split}_counts.txt"
done

wc -l "$OUT/train.list" "$OUT/val.list" | tee "$OUT/list_counts.txt"

lstmtraining \
  --continue_from "$OUT/rus.lstm" \
  --traineddata "$TESSDATA/rus.traineddata" \
  --model_output "$OUT/${MODEL_NAME}" \
  --train_listfile "$OUT/train.list" \
  --eval_listfile "$OUT/val.list" \
  --max_iterations "$MAX_ITERATIONS" \
  --debug_interval -1 \
  > "$OUT/lstmtraining.log" 2>&1

checkpoint="$(find "$OUT" -maxdepth 1 -name "${MODEL_NAME}_checkpoint" -o -name "${MODEL_NAME}*.checkpoint" | sort | tail -1)"
if [[ -z "$checkpoint" ]]; then
  echo "No checkpoint produced" >&2
  exit 1
fi

lstmtraining \
  --stop_training \
  --continue_from "$checkpoint" \
  --traineddata "$TESSDATA/rus.traineddata" \
  --model_output "$OUT/${MODEL_NAME}.traineddata" \
  > "$OUT/stop_training.log" 2>&1

echo "traineddata=$OUT/${MODEL_NAME}.traineddata"
