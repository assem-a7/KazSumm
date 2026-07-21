# KazSumm: A Benchmark Dataset for Abstractive Summarization of Kazakh News Texts

KazSumm is the first publicly available benchmark dataset for Kazakh news headline generation (abstractive summarization). It consists of news article–headline pairs collected from Tengrinews.kz.

## Contents

| File | Description | Records |
|------|-------------|---------|
| `kazsum_full.zip` | Main dataset after the full filtering pipeline | 26,273 |
| `kazsum_unfiltered.zip` | Alternative version that preserves short (<5-word) headlines | 27,485 |
| `train.test.val.zip` | Train / validation / test split used in the paper | 21,018 / 2,627 / 2,628 |

Each CSV file contains two columns:
- `text` — the full body of the news article
- `title` — the corresponding headline (the summarization target)

## Dataset Construction

The raw data was collected from Tengrinews.kz. A five-stage filtering pipeline was applied to remove articles with very short bodies (<50 words), overly short (<5-word) or long (>25-word) headlines, portal branding in the headline, and cases where the headline duplicates the body. The `kazsum_unfiltered` version restores the 897 records removed by the short-headline filter, since in the agglutinative Kazakh language a four-word headline can be fully informative.

## Splits

The dataset is split 80/10/10 into training, validation, and test sets using a fixed random seed (42) for reproducibility. All experiments in the accompanying paper use the same split provided in `train.test.val.zip`.

## Usage

```python
import pandas as pd
# unzip first, then:
df = pd.read_csv("train.csv")
print(df[["text", "title"]].head())
```

## License

This dataset is released under the MIT License. The original news data from Tengrinews.kz is used for research purposes.

## Citation

If you use KazSumm in your research, please cite:
[Citation will be added upon publication]

## Contact

For questions, please contact: asema.abdikarim@gmail.com
