#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"
python prepare_representations.py
for d in time frequency; do
  for m in 1D-CNN CNN-LSTM CNN-GRU Light-DSCNN; do
    python train_models.py --domain "$d" --model "$m"
  done
done
python postprocess_results.py
python rev2_init_sensitivity.py
python rev2_conflict_freq.py
python rev2_near_freq.py
python timing.py
printf '%s\n' 'Reproduction run complete.'
