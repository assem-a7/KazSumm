"""
evaluate.py
===========
Score model predictions with ROUGE and BERTScore, and reproduce the paper's
statistical tables:

  * Table II   — point estimates (ROUGE-1/2/L, BERTScore-F1) per model
  * Table IIa  — 95% bootstrap confidence intervals (B = 10,000, seed 42)
  * Table IIb  — paired bootstrap significance tests between selected models

Metric settings match the paper:
    ROUGE      : rouge_score, use_stemmer=False, reported as F-measure x 100
    BERTScore  : xlm-roberta-large, num_layers=17, rescale_with_baseline=False

Inputs are the prediction files produced by inference.py / textrank_baseline.py,
each a JSON of the form {"preds": [...], "refs": [...]}. All models are assumed
to share the same test references.

Usage
-----
    python src/evaluate.py --preds_dir preds --out_dir results

Expected files in --preds_dir (missing ones are skipped):
    fullft_preds.json, lora_r8_preds.json, lora_r16_preds.json,
    textrank_preds.json, zeroshot_preds.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# key -> (prediction filename, human-readable display name)
MODEL_FILES = {
    'mT5_FullFT':   ('fullft_preds.json',   'mT5 full FT'),
    'LoRA_r16':     ('lora_r16_preds.json', 'mT5 + LoRA r=16'),
    'LoRA_r8':      ('lora_r8_preds.json',  'mT5 + LoRA r=8'),
    'TextRank':     ('textrank_preds.json', 'TextRank'),
    'mT5_ZeroShot': ('zeroshot_preds.json', 'mT5 zero-shot'),
}

# Pairwise comparisons reported in Table IIb.
COMPARISONS = [
    ('mT5_FullFT', 'LoRA_r16'),
    ('mT5_FullFT', 'LoRA_r8'),
    ('mT5_FullFT', 'TextRank'),
    ('LoRA_r16', 'TextRank'),
]

METRIC_NAMES = {'r1': 'ROUGE-1', 'r2': 'ROUGE-2', 'rL': 'ROUGE-L', 'bert': 'BERTScore-F1'}


def per_example_scores(preds, refs, device):
    """Return per-example arrays for ROUGE-1/2/L (F) and BERTScore-F1."""
    from rouge_score import rouge_scorer
    from bert_score import score as bertscore_fn

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        s = rouge.score(r, p)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rL.append(s['rougeL'].fmeasure)

    _, _, f1 = bertscore_fn(preds, refs, model_type="xlm-roberta-large", num_layers=17,
                            batch_size=16, rescale_with_baseline=False, verbose=False,
                            device=device)
    return {'r1': np.array(r1), 'r2': np.array(r2), 'rL': np.array(rL),
            'bert': f1.numpy()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score predictions: ROUGE, BERTScore, CIs, significance.")
    ap.add_argument('--preds_dir', default='preds', help='Directory with *_preds.json files.')
    ap.add_argument('--out_dir', default='results', help='Directory for the output tables.')
    ap.add_argument('--bootstrap', type=int, default=10000, help='Number of bootstrap resamples.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cpu'

    # ---- load predictions and compute per-example metrics --------------------
    per_example, refs_ref = {}, None
    for key, (fname, _) in MODEL_FILES.items():
        path = os.path.join(args.preds_dir, fname)
        if not os.path.exists(path):
            print(f"skip {key}: {path} not found")
            continue
        d = json.load(open(path, encoding='utf-8'))
        preds, refs = d['preds'], d['refs']
        refs_ref = refs_ref or refs
        print(f"scoring {key} (n={len(preds)}) ...")
        per_example[key] = per_example_scores(preds, refs, device)

    if not per_example:
        raise SystemExit("No prediction files found in --preds_dir.")

    n = len(refs_ref)
    rng = np.random.default_rng(args.seed)
    boot_idx = [rng.choice(n, n, replace=True) for _ in range(args.bootstrap)]

    def ci95(scores):
        means = np.array([scores[bi].mean() for bi in boot_idx])
        return scores.mean() * 100, np.percentile(means, 2.5) * 100, np.percentile(means, 97.5) * 100

    def paired_test(a, b):
        diffs = np.array([a[bi].mean() - b[bi].mean() for bi in boot_idx])
        obs = (a.mean() - b.mean()) * 100
        lo, hi = np.percentile(diffs, 2.5) * 100, np.percentile(diffs, 97.5) * 100
        p = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0)
        return obs, lo, hi, p

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Table IIa: point estimates + 95% CI --------------------------------
    ci_rows = []
    for key in [k for k in MODEL_FILES if k in per_example]:
        row = {'Model': MODEL_FILES[key][1]}
        for m in ['r1', 'r2', 'rL', 'bert']:
            mean, lo, hi = ci95(per_example[key][m])
            row[METRIC_NAMES[m]] = f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"
        ci_rows.append(row)
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(f'{args.out_dir}/table_ci.csv', index=False, encoding='utf-8-sig')
    print("\n=== Table IIa: 95% confidence intervals ===")
    print(ci_df.to_string(index=False))

    # ---- Table IIb: pairwise significance -----------------------------------
    sig_rows = []
    for a, b in COMPARISONS:
        if a not in per_example or b not in per_example:
            continue
        row = {'Comparison': f"{MODEL_FILES[a][1]} vs. {MODEL_FILES[b][1]}"}
        pmax = 0.0
        for m in ['r1', 'r2', 'rL', 'bert']:
            obs, lo, hi, p = paired_test(per_example[a][m], per_example[b][m])
            row[f'Δ{METRIC_NAMES[m]}'] = f"{obs:+.2f}"
            pmax = max(pmax, p)
        row['p-value'] = "< 0.001" if pmax < 0.001 else f"{pmax:.3f}"
        sig_rows.append(row)
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(f'{args.out_dir}/table_significance.csv', index=False, encoding='utf-8-sig')
    print("\n=== Table IIb: pairwise significance (paired bootstrap) ===")
    print(sig_df.to_string(index=False))

    print(f"\nSaved: {args.out_dir}/table_ci.csv, {args.out_dir}/table_significance.csv")


if __name__ == '__main__':
    main()
