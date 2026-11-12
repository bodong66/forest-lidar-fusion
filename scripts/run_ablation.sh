#!/usr/bin/env bash
# Species-classification ablation: modality and fusion strategy, same seed and schedule.
# Usage: bash scripts/run_ablation.sh configs/species_forspecies20k.yaml [extra overrides...]
set -euo pipefail
CFG=${1:-configs/species_forspecies20k.yaml}
shift || true

python -m forestfusion.train_species --config "$CFG" model.modality=points  output_dir=runs/species_points  "$@"
python -m forestfusion.train_species --config "$CFG" model.modality=views   output_dir=runs/species_views   "$@"
for F in concat gated attention; do
  python -m forestfusion.train_species --config "$CFG" model.modality=fusion model.fusion=$F \
    output_dir=runs/species_fusion_$F "$@"
done
# Does modality dropout buy robustness? Same best fusion, without it.
python -m forestfusion.train_species --config "$CFG" model.modality=fusion model.fusion=attention \
  model.modality_dropout=0.0 output_dir=runs/species_fusion_attention_nodrop "$@"

python scripts/summarize_runs.py runs/species_* --metric val_balanced_acc
