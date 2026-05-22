# Artifacts

This directory contains files that are required at runtime but are too large or too data-specific to keep inside source modules.

Expected layout:

```text
artifacts/
  models/
    rfdetr_small_price_tag_all_annotated_tiled1280_e8_checkpoint_best_total.pth
  data/
    db_hack.csv
    sample.csv
  special_symbol_templates/
    full_tags/
      track_*.jpg
  training_data/
    # optional annotated training videos and annotations
```

The current package includes the detector checkpoint, `db_hack.csv`, `sample.csv`, and the small set of full-tag template images used by the special-symbol classifier.

Training data is not bundled by default. Put annotated videos/metadata into `artifacts/training_data` or pass `-DataRoot` to `pipeline/run_train_rfdetr.ps1`.
