"""
textrank_baseline.py
====================
Extractive TextRank baseline (single-sentence headline) used in the paper.

For every article the graph-based TextRank summariser (``sumy``) selects the one
most central sentence as the predicted headline. If TextRank fails on a row, the
first sentence of the body is used as a fallback.

Note: following the paper's pipeline, the sentence tokenizer is initialised with
the ``"english"`` model — Kazakh is not supported by ``sumy`` directly, and this
choice is kept for exact reproducibility of the reported numbers.

Usage
-----
    python src/textrank_baseline.py --test_csv data/test.csv --out preds/textrank_preds.json

Output: a JSON file {"preds": [...], "refs": [...]} consumed by evaluate.py.
"""

import argparse
import json
import os
import re

import pandas as pd

WS = lambda t: ' '.join(str(t).strip().split())


def make_summarizer():
    """Return a function text -> one-sentence extractive summary (TextRank)."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    import nltk

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    summarizer = TextRankSummarizer()

    def summarize(text: str) -> str:
        try:
            parser = PlaintextParser.from_string(str(text), Tokenizer("english"))
            summary = summarizer(parser.document, 1)  # 1 sentence
            result = " ".join(str(s) for s in summary).strip()
            if not result:
                raise ValueError("empty summary")
            return result
        except Exception:
            sents = re.split(r'[.!?]', str(text))
            return sents[0].strip() if sents else str(text)[:100]

    return summarize


def main() -> None:
    ap = argparse.ArgumentParser(description="TextRank extractive baseline.")
    ap.add_argument('--test_csv', required=True, help='Test CSV with columns title, text.')
    ap.add_argument('--out', default='preds/textrank_preds.json', help='Output predictions JSON.')
    args = ap.parse_args()

    test_df = pd.read_csv(args.test_csv)
    summarize = make_summarizer()

    preds = [summarize(t) for t in test_df['text']]
    refs = [WS(t) for t in test_df['title']]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'preds': preds, 'refs': refs}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(preds)} TextRank predictions -> {args.out}")


if __name__ == '__main__':
    main()
