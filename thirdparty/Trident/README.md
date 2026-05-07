# Trident Vendor Subset

This directory contains the minimal Trident runtime used by FreeOcc for open-vocabulary segmentation.

The original Trident benchmark scripts, demo assets, dataset conversion scripts, and MMSeg benchmark configs were removed because this project instantiates `Trident` directly from `src/gaussian_mapping.py`.

Kept runtime components:

- `trident.py`
- `open_clip/`
- `segment_anything/`
- `seg_utils/`
- `prompts/`
- `pamr.py`
- `myutils.py`

The SAM checkpoint is expected to be provided by the main project at `pretrained/sam_vit_b_01ec64.pth`.

Original project: https://github.com/YuHengsss/Trident
