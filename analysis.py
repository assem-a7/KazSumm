"""
analysis.py
===========
Two qualitative analyses reported in the paper:

  1. Language identification (Section on zero-shot behaviour).
     mT5-XLSum has no Kazakh training data and tends to generate Kyrgyz. This
     runs langdetect (deterministic, seed 42) over a predictions file and reports
     the distribution of detected languages, flagging empty / degenerate outputs
     (including leftover mT5 sentinel tokens).

  2. Human error annotation (error-analysis table).
     Aggregates the manual annotation sheet: per-model counts of each error type
     and mean fluency / adequacy. Any column whose name starts with 'error_' is
     treated as a binary error flag, so the script adapts to the sheet's columns.

Usage
-----
    # Language ID on the zero-shot predictions
    python src/analysis.py --langid_preds preds/zeroshot_preds.json

    # Aggregate the human error-annotation sheet
    python src/analysis.py --annotations data/error_annotation_sheet.csv

Both flags can be given together. The annotation CSV is expected to contain the
columns: model, fluency_1to5, adequacy_1to5, and one or more error_* flags.
"""

import argparse
import json
import re

import pandas as pd

WS = lambda t: re.sub(r'\s+', ' ', str(t).strip())


# --------------------------------------------------------------------------- #
# 1. Language identification
# --------------------------------------------------------------------------- #
def language_distribution(preds, name="predictions"):
    """Return the detected-language distribution (%) for a list of predictions."""
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 42

    labels = []
    for p in preds:
        p = WS(p)
        # empty / degenerate outputs, including mT5 sentinel tokens
        if len(p) < 3 or 'extra_id' in p or len(re.findall(r'\w+', p)) < 2:
            labels.append('empty/degenerate')
            continue
        try:
            labels.append(detect_langs(p)[0].lang)
        except Exception:
            labels.append('undetected')

    counts = pd.Series(labels).value_counts()
    pct = (counts / len(labels) * 100).round(2)
    print(f"\n--- Language distribution: {name} (n={len(labels)}) ---")
    print(pct.to_string())
    return pct.to_dict()


# --------------------------------------------------------------------------- #
# 2. Human error annotation
# --------------------------------------------------------------------------- #
def aggregate_annotations(csv_path):
    """Per-model error-type counts and mean fluency / adequacy from the sheet."""
    df = pd.read_csv(csv_path)
    error_cols = [c for c in df.columns if c.lower().startswith('error_')]
    print(f"\n--- Error annotation ({len(df)} rows) ---")
    print(f"Error columns: {error_cols}")

    rows = []
    for model, g in df.groupby('model'):
        row = {'model': model, 'n': len(g)}
        for c in error_cols:
            k = int(g[c].fillna(0).astype(float).sum())
            row[c] = f"{k} ({k / len(g) * 100:.0f}%)"
        for col in ('fluency_1to5', 'adequacy_1to5'):
            if col in g.columns:
                row[col.replace('_1to5', '')] = round(g[col].astype(float).mean(), 2)
        rows.append(row)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Language-ID and error-annotation analyses.")
    ap.add_argument('--langid_preds', default=None, help='Predictions JSON for language ID.')
    ap.add_argument('--annotations', default=None, help='Human error-annotation CSV.')
    ap.add_argument('--out_dir', default='results')
    args = ap.parse_args()

    if not args.langid_preds and not args.annotations:
        ap.error("Provide --langid_preds and/or --annotations.")

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    if args.langid_preds:
        d = json.load(open(args.langid_preds, encoding='utf-8'))
        dist = language_distribution(d['preds'], name=os.path.basename(args.langid_preds))
        json.dump(dist, open(f'{args.out_dir}/langid.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        print(f"Saved {args.out_dir}/langid.json")

    if args.annotations:
        out = aggregate_annotations(args.annotations)
        out.to_csv(f'{args.out_dir}/error_summary.csv', index=False, encoding='utf-8-sig')
        print(f"Saved {args.out_dir}/error_summary.csv")


if __name__ == '__main__':
    main()
