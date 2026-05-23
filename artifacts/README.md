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

Training data is not bundled by default. Put annotated folders into `artifacts/training_data` or mount another folder through `HOST_TRAINING_DATA_DIR`.

Only folders with matching `<folder>.mp4` and `<folder>.csv` files are used for detector training. Folders without annotations, such as `Unlabeled`, are ignored by the dataset builder.

Training outputs are written to `runtime/training/...`; trained checkpoints are not written back into `artifacts/models` automatically.
