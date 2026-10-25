# KazSumm — Kazakh News Headline Generation

Research code for generating short Kazakh headlines from news articles and comparing **full mT5 fine-tuning** with **LoRA adaptation**.

The repository includes dataset archives, preprocessing, a TextRank baseline, training, inference and evaluation scripts. It is a collaborative research project; contributors are credited in [LICENSE](LICENSE).

## What this project explores

Can a small set of trainable LoRA parameters provide a useful alternative to full fine-tuning for Kazakh headline generation?

The implemented comparison uses `csebuetnlp/mT5_multilingual_XLSum` as the base model, full fine-tuning, LoRA ranks 8 and 16, and an extractive TextRank baseline. The target is the original news headline, rather than a human-written multi-sentence summary.

**Current status:** this repository contains code and data archives. Trained checkpoints, prediction files, evaluation tables, the manuscript and human annotation sheets are not included in the current tree. No model-quality numbers are asserted here.

## Dataset

The news articles originate from Tengrinews.kz. Each example has `text` (article body) and `title` (reference headline).

| Archive | Contents described by the original project documentation |
| --- | --- |
| [kazsum_full.zip](kazsum_full.zip) | Main filtered dataset: 26,273 pairs |
| [kazsum_unfiltered.zip](kazsum_unfiltered.zip) | Alternative dataset retaining short headlines: 27,622 pairs |
| [train.test.val.zip](train.test.val.zip) | Fixed train / validation / test CSVs: 21,018 / 2,627 / 2,628 rows |

These counts are retained from the existing documentation; archive contents have not been independently recounted in this documentation update.

[preprocessing.py](preprocessing.py) filters short bodies, very short or long headlines, portal branding and body–headline duplicates, then removes boilerplate and splits rows 80/10/10 with seed 42. The “unfiltered” variant skips only the short-headline filter; it still applies the other filters.

## Start here

Use Python 3.10 in a virtual environment. A compatible GPU is recommended for neural training. Model weights and NLTK resources require internet access on first use.

```bash
git clone https://github.com/assem-a7/KazSumm.git
cd KazSumm
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux, then install:

```bash
python -m pip install -r requirements.txt
```

The dependency file uses version ranges and unpinned packages, rather than a complete lockfile. This documentation update has not validated a fresh training environment.

Extract `train.test.val.zip` and arrange the split files as `data/train.csv`, `data/val.csv` and `data/test.csv`. You can inspect archive member names first:

```bash
python -m zipfile -l train.test.val.zip
```

For a baseline without neural fine-tuning:

```bash
python textrank_baseline.py --test_csv data/test.csv --out preds/textrank_preds.json
```

## Train and generate

The default input and target limits are 512 and 84 tokens. Training runs for up to three epochs with early stopping on validation loss. Precision is bf16 when supported by the GPU, otherwise fp32.

| Setting | Full fine-tuning | LoRA |
| --- | --- | --- |
| Learning rate | 5e-4 | 3e-4 |
| Trainable weights | All model parameters | Adapter parameters |
| Adapter configuration | — | Rank 8 or 16; alpha 32; dropout 0.1; target modules q and v |
| Batch / gradient accumulation | 4 / 4 | 4 / 4 |
| Optimizer / seed | Adafactor / 42 | Adafactor / 42 |

```bash
python train.py --mode full --data_dir data --out checkpoints/mt5_full_ft
python train.py --mode lora --lora_r 8 --data_dir data --out checkpoints/mt5_lora_r8
python train.py --mode lora --lora_r 16 --data_dir data --out checkpoints/mt5_lora_r16

python inference.py --model_path checkpoints/mt5_full_ft/best_model --test_csv data/test.csv --out preds/fullft_preds.json
python inference.py --model_path checkpoints/mt5_lora_r8/best_model --test_csv data/test.csv --out preds/lora_r8_preds.json
python inference.py --model_path checkpoints/mt5_lora_r16/best_model --test_csv data/test.csv --out preds/lora_r16_preds.json
```

Inference automatically distinguishes full checkpoints from LoRA adapters. Generation uses four beams, a maximum of 84 tokens, no repeated bigrams and length penalty 0.6.

To rebuild the dataset from a separately supplied raw CSV with `title` and `text` columns:

```bash
python -c "from pathlib import Path; Path('data').mkdir(exist_ok=True)"
python preprocessing.py --raw tengri_news.csv --outdir data --variant filtered
```

## Evaluation and current limitations

[evaluate.py](evaluate.py) computes ROUGE and BERTScore, bootstrap confidence intervals and paired comparisons. It expects prediction JSON files containing aligned `preds` and `refs` arrays.

**ROUGE needs correction before its scores can be interpreted for Kazakh.** The current evaluator uses the default `rouge_score` tokenizer, whose [implementation](https://github.com/google-research/google-research/blob/master/rouge/tokenize.py) retains Latin a–z and digits, discarding Kazakh Cyrillic letters. A Unicode-aware tokenizer and a fresh evaluation are required.

Additional limitations to address:

- TextRank uses an English sentence tokenizer. Its suitability for Kazakh needs checking.
- Inference catches generation failures and substitutes the first 100 characters of the input without recording a failure flag.
- The evaluator assumes prediction lengths and reference order agree across models; it does not validate that alignment.
- The random split does not perform explicit cross-split duplicate or near-duplicate checks.
- Human annotation aggregation is implemented in [analysis.py](analysis.py), but the annotation sheet is not included.

After correcting and validating evaluation, the command below writes `results/table_ci.csv` and `results/table_significance.csv`. It skips missing prediction files.

```bash
python evaluate.py --preds_dir preds --out_dir results
```

## Code guide

| File | Purpose |
| --- | --- |
| [preprocessing.py](preprocessing.py) | Cleaning, filtering and seeded splits |
| [textrank_baseline.py](textrank_baseline.py) | Single-sentence extractive baseline |
| [train.py](train.py) | Full fine-tuning and LoRA training |
| [inference.py](inference.py) | Headline generation and prediction export |
| [evaluate.py](evaluate.py) | Metrics and bootstrap comparisons |
| [analysis.py](analysis.py) | Language-ID diagnostics and annotation aggregation |

## Attribution and licensing

See [LICENSE](LICENSE) for contributor credits, MIT code terms and the separate notice concerning the underlying news articles. Article copyrights remain with the original rightsholders; the code license does not establish unrestricted rights to the news texts.

**Tools:** Python, PyTorch, Hugging Face Transformers, PEFT, pandas, scikit-learn.

