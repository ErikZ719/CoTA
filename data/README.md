# Data

Image lists only; the images are COCO val2014 (`http://images.cocodataset.org/zips/val2014.zip`), expected under
`$COTA_ROOT/coco2014/val2014/`. Copy these files to `$COTA_EXP/data/` for the drivers and aggregators.

| File | What |
|---|---|
| `coco500_final.json` | the 500 COCO val2014 images of every Repeat-Curse table (`files`: image names; `criterion`: how the list was built, 485 images with a repeat-free uncached response at L=128 plus 15 with a benign pair, one image replaced on 2026-09-05) |
| `coco500_seed42.json`, `coco500_batch2_seed43.json` | the two 500-image pools the list was drawn from (`eval/sample_coco500.py`) |
| `coco100_seed0.json` | the 100-image diagnosis sample of the findings analyses (94 repeat under dLLM-Cache); the first 100 images of `coco500_final.json` are the sample of the design-alternative experiments |
| `probe_*.json`, `supplement_262376.json` | single-image lists for the decode-row probes and the replacement image |

The judge (`judge/judge_table6.py`) also needs the five human captions of each evaluation image as
`{image file name: [captions]}`, built from the COCO 2014 caption annotations.
