"""
preprocessing.py
================
Build the KazSumm dataset from the raw Tengrinews.kz dump and produce the exact
train / validation / test split used in the paper.

Pipeline (five-stage filter + regex cleaning), reproducing the paper:
    F1  remove articles whose body has  < 50 words
    F2  remove headlines with          < 5 words   (skipped for the "unfiltered" variant)
    F3  remove headlines with          > 25 words
    F4  remove portal branding in the headline (Tengrinews / Tengritravel)
    F5  remove rows where the body equals the headline
    then regex-clean body/title and re-apply F1 on the cleaned body.

The dataset is split 80 / 10 / 10 with a fixed seed (42) for reproducibility.

Usage
-----
    # Main filtered dataset (26,273 pairs) + 21,018 / 2,627 / 2,628 split
    python src/preprocessing.py --raw tengri_news.csv --outdir data --variant filtered

    # Alternative dataset that keeps short (<5-word) headlines
    python src/preprocessing.py --raw tengri_news.csv --outdir data --variant unfiltered

The raw file must contain the columns: title, text (extra columns are ignored).
Outputs (in --outdir): kazsum_full.csv, train.csv, val.csv, test.csv
(for --variant unfiltered: kazsum_unfiltered.csv, train_unfilt.csv, ...).
"""

import argparse
import io
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Remove Tengrinews boilerplate, calls-to-subscribe, and URLs from a body."""
    text = str(text)
    text = re.sub(r',?\s*-?\s*деп хабарлайды[\s\w\.«»""]+тілшісі[^\.]*\.?', '', text)
    text = re.sub(r'Tengri\w*\.kz\s+тілшісі', '', text)
    text = re.sub(r'Сілтемесіз жаңалық оқисыз ба[^$]*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Telegram желісінде бізге жазыл[^$]*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_title(title: str) -> str:
    """Normalise whitespace and strip a trailing '- БАҚ' tag from a headline."""
    title = str(title)
    title = re.sub(r'\s+', ' ', title).strip()
    title = re.sub(r'\s*[-–—]\s*БАҚ\s*$', '', title)
    return title


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a CSV robustly: strip BOM and try utf-8 (with/without BOM) then cp1251."""
    for enc in ('utf-8-sig', 'utf-8', 'cp1251'):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError("Could not read CSV in any of the tried encodings.")


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def build_dataset(df_raw: pd.DataFrame, keep_short_headlines: bool = False) -> pd.DataFrame:
    """Apply the five-stage filter + cleaning and return the final dataframe.

    Parameters
    ----------
    df_raw : DataFrame with at least the columns ``title`` and ``text``.
    keep_short_headlines : if True, the F2 filter (title < 5 words) is skipped,
        which restores the short-headline records ("unfiltered" variant).
    """
    df = df_raw[['title', 'text']].copy()
    n0 = len(df)
    df = df.dropna(subset=['title', 'text'])
    df['tw'] = df['text'].apply(lambda x: len(str(x).split()))
    df['hw'] = df['title'].apply(lambda x: len(str(x).split()))

    funnel = {'Raw': n0, 'NaN title/text': n0 - len(df)}

    n = len(df); df = df[df['tw'] >= 50]; funnel['F1: text < 50 words'] = n - len(df)
    if not keep_short_headlines:
        n = len(df); df = df[df['hw'] >= 5]; funnel['F2: title < 5 words'] = n - len(df)
    n = len(df); df = df[df['hw'] <= 25]; funnel['F3: title > 25 words'] = n - len(df)
    n = len(df)
    df = df[~df['title'].str.contains('Tengrinews|tengrinews|Tengritravel', na=False)]
    funnel['F4: brand in title'] = n - len(df)
    n = len(df); df = df[df['text'].str.strip() != df['title'].str.strip()]
    funnel['F5: text == title'] = n - len(df)

    # Regex clean, then re-apply the body-length filter on the cleaned text.
    df['text'] = df['text'].apply(clean_text)
    df['title'] = df['title'].apply(clean_title)
    df['tw'] = df['text'].apply(lambda x: len(x.split()))
    n = len(df); df = df[df['tw'] >= 50].reset_index(drop=True)
    funnel['Re-F1: text < 50 (after clean)'] = n - len(df)
    funnel['FINAL'] = len(df)

    print("\nFiltering funnel:")
    for k, v in funnel.items():
        print(f"   {k:<34}: {v:>7,}")
    return df[['title', 'text']].reset_index(drop=True)


def split_and_save(df: pd.DataFrame, outdir: str, suffix: str = "") -> None:
    """80/10/10 split with a fixed seed and save all CSVs as utf-8-sig."""
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    os.makedirs(outdir, exist_ok=True)
    train_df.to_csv(f'{outdir}/train{suffix}.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv(f'{outdir}/val{suffix}.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv(f'{outdir}/test{suffix}.csv', index=False, encoding='utf-8-sig')
    print(f"\nSaved split (utf-8-sig): "
          f"train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build KazSumm and produce the paper split.")
    ap.add_argument('--raw', required=True, help='Path to the raw Tengrinews CSV (title, text).')
    ap.add_argument('--outdir', default='data', help='Output directory for the CSV files.')
    ap.add_argument('--variant', choices=['filtered', 'unfiltered'], default='filtered',
                    help="'filtered' = main dataset; 'unfiltered' keeps short (<5-word) headlines.")
    args = ap.parse_args()

    df_raw = read_csv_robust(args.raw)
    print(f"Raw rows: {len(df_raw):,} | columns: {df_raw.columns.tolist()}")

    keep_short = args.variant == 'unfiltered'
    df = build_dataset(df_raw, keep_short_headlines=keep_short)

    if args.variant == 'filtered':
        df.to_csv(f'{args.outdir}/kazsum_full.csv', index=False, encoding='utf-8-sig')
        split_and_save(df, args.outdir, suffix="")
    else:
        df.to_csv(f'{args.outdir}/kazsum_unfiltered.csv', index=False, encoding='utf-8-sig')
        split_and_save(df, args.outdir, suffix="_unfilt")

    print(f"\nDone. Final dataset size ({args.variant}): {len(df):,}")


if __name__ == '__main__':
    main()
