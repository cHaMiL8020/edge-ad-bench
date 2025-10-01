# IMS Center Dataset (Set 2) — How We Use It
- Raw ASCII files under `data/raw/ims_center/set2/*`, 1-second each (~20,480 samples).
- Time-based split: 70% train, 15% val, 15% test.
- Labeling: `last_k` positives to keep train/val healthy and test mixed.
- Windowing: split by file first, then window within each split to avoid leakage.

## Commands (non-windowed)
python -m bench.cli action=prepare dataset=ims_center \
  dataset.window.enabled=false \
  dataset.labeling.mode=last_k +dataset.labeling.last_k=120
python -m bench.cli action=train dataset=ims_center model=iforest
python -m bench.cli action=eval  dataset=ims_center model=iforest

## Commands (windowed)
python -m bench.cli action=prepare dataset=ims_center \
  dataset.window.enabled=true dataset.window.length=2048 dataset.window.stride=1024 \
  dataset.labeling.mode=last_k +dataset.labeling.last_k=120
python -m bench.cli action=train dataset=ims_center model=iforest
python -m bench.cli action=eval  dataset=ims_center model=iforest
