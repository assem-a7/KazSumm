# KazSumm: Benchmarking Low-Rank Adaptation for Kazakh Abstractive Summarization

This repository accompanies the paper *"KazSumm: Benchmarking Low-Rank Adaptation
for Kazakh Abstractive Summarization"*. It contains the KazSumm dataset, the fixed
train/validation/test split, and the full code needed to reproduce every result,
table, and analysis in the paper.

## Dataset

KazSumm is a Kazakh news headline-generation dataset built from Tengrinews.kz
articles. Two versions are provided:

| File | Pairs | Description |
|------|------:|-------------|
| `kazsum_full.csv` | 26,273 | Main filtered dataset (five-stage filter + cleaning) |
| `kazsum_unfiltered.csv` | 27,622 | Alternative version that keeps short (<5-word) headlines |

Each file has two columns: `title` (the headline / reference summary) and `text`
(the article body). The fixed split of the main dataset is:

| Split | Rows |
|-------|-----:|
| train | 21,018 |
| validation | 2,627 |
| test | 2,628 |

The split is produced with an 80/10/10 ratio and a fixed random seed (42), and is
released as `train.csv`, `val.csv`, `test.csv` so the exact partition can be
reproduced without rerunning the split.

### Data provenance, license, and redistribution

The article texts originate from **Tengrinews.kz**. Copyright of the original
article texts remains with the publisher. This dataset is released for
**non-commercial academic research only**, and redistribution here is limited to
what is necessary to reproduce the experiments reported in the paper. Anyone
intending to redistribute the data or use it commercially should obtain permission
from the original publisher.

The **code, the filtering/splitting pipeline, and the human error annotations** in
this repository are released under the MIT License (see `LICENSE`).

## Repository structure

```
KazSumm/
├── README.md
├── LICENSE                     # MIT (code, pipeline, annotations)
├── requirements.txt            # pinned dependencies
├── preprocessing.py            # raw Tengrinews CSV -> filtered dataset + split
├── textrank_baseline.py        # extractive TextRank baseline
├── train.py                    # full fine-tuning and LoRA (r = 8, 16)
├── inference.py                # generate headlines (full model or LoRA adapter)
├── evaluate.py                 # ROUGE + BERTScore + 95% CIs + significance
├── analysis.py                 # language-ID + human error-annotation aggregation
├── kazsum_full.zip             # main filtered dataset (26,273)
├── kazsum_unfiltered.zip       # alternative version (27,622)
└── train.test.val.zip          # fixed train / val / test split
```

Trained LoRA adapters (about 3.4 MB and 6.8 MB) and the full fine-tuned checkpoint
can be regenerated with `train.py`; adapters are small enough to be released
alongside the code if desired.

## Setup

```bash
pip install -r requirements.txt
```

Experiments were run on Google Colab with a single Tesla T4 GPU. mT5 is unstable in
fp16, so training uses **bf16** where supported and falls back to fp32.

## Model and hyperparameters

All fine-tuned models start from **`csebuetnlp/mT5_multilingual_XLSum`** (mT5-base,
12 encoder/decoder layers, d_model 768).

| Setting | Full fine-tuning | LoRA (r = 8 / 16) |
|---------|------------------|--------------------|
| Learning rate | 5e-4 | 3e-4 |
| LoRA alpha / dropout / targets | — | 32 / 0.1 / q, v |
| Trainable parameters | 100% | ~0.15% / ~0.30% |
| Batch size (per device x grad. accum.) | 4 x 4 | 4 x 4 |
| Epochs (early stopping, patience 2) | 3 | 3 |
| Warmup steps / weight decay | 200 / 0.01 | 200 / 0.01 |
| Optimizer / scheduler | Adafactor / linear | Adafactor / linear |
| Precision / seed | bf16 / 42 | bf16 / 42 |

Generation (all models): beam search with 4 beams, `no_repeat_ngram_size = 2`,
max input 512 tokens, max output 84 tokens, length penalty 0.6.

Evaluation: ROUGE-1/2/L (`rouge_score`, `use_stemmer=False`) and BERTScore-F1
(`xlm-roberta-large`, `num_layers=17`, `rescale_with_baseline=False`). Confidence
intervals and significance tests use bootstrap resampling (B = 10,000, seed 42).

## Reproducing the results

```bash
# 1. Build the dataset and split from the raw dump (already provided as CSVs/zips)
python preprocessing.py --raw tengri_news.csv --outdir data --variant filtered

# 2. Baselines and models
python textrank_baseline.py --test_csv data/test.csv --out preds/textrank_preds.json
python train.py --mode full  --data_dir data --out checkpoints/mt5_full_ft
python train.py --mode lora --lora_r 8  --data_dir data --out checkpoints/mt5_lora_r8
python train.py --mode lora --lora_r 16 --data_dir data --out checkpoints/mt5_lora_r16

# 3. Generate predictions on the test set
python inference.py --model_path checkpoints/mt5_full_ft/best_model  --test_csv data/test.csv --out preds/fullft_preds.json
python inference.py --model_path checkpoints/mt5_lora_r8/best_model  --test_csv data/test.csv --out preds/lora_r8_preds.json
python inference.py --model_path checkpoints/mt5_lora_r16/best_model --test_csv data/test.csv --out preds/lora_r16_preds.json

# 4. Scores, confidence intervals, and significance tests
python evaluate.py --preds_dir preds --out_dir results

# 5. Qualitative analyses
python analysis.py --langid_preds preds/zeroshot_preds.json
python analysis.py --annotations data/error_annotation_sheet.csv
```

## Citation

```bibtex
@article{kazsumm2026,
  title   = {KazSumm: Benchmarking Low-Rank Adaptation for Kazakh Abstractive Summarization},
  author  = {Abdikarim, Asem and others},
  year    = {2026},
  note    = {Citation details will be updated upon publication.}
}
```
