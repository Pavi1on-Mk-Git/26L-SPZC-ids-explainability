#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

for lr in 0.01 0.005 0.001 0.0005 0.0001; do
    echo "=== Running experiment with learning rate ${lr} ==="
    pdm run python src/scripts/experiments.py \
        --no-save-processed \
        --oracle-only \
        --learning-rate "${lr}"
done

for monitor in val_loss train_loss val_acc train_acc; do
    echo "=== Running experiment with early stopping monitor ${monitor} ==="
    pdm run python src/scripts/experiments.py \
        --no-save-processed \
        --oracle-only \
        --early-stopping-monitor "${monitor}"
done


echo "=== Running experiment with explainer ==="
pdm run python src/scripts/experiments.py \
    --no-save-processed \
    --early-stopping-monitor "val_acc"

echo "=== Running experiment with explainer on CSE-CIC-IDS-2018 ==="
pdm run python src/scripts/experiments.py \
    --dataset CSE-CIC-IDS-2018 \
    --no-save-processed \
    --early-stopping-monitor "val_acc"
