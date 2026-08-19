"""
inference.py
============
Generate Kazakh headlines with a trained model and save the predictions.

Works with both kinds of checkpoint produced by train.py:
  * a full fine-tuned model directory, or
  * a LoRA adapter directory (auto-detected via adapter_config.json; the base
    model is read from the adapter config, or can be overridden with --base_model).

Generation settings match the paper: beam search with 4 beams,
no_repeat_ngram_size = 2, max length 84, length penalty 0.6.

Usage
-----
    # LoRA adapter
    python src/inference.py --model_path checkpoints/mt5_lora_r16/best_model \
        --test_csv data/test.csv --out preds/lora_r16_preds.json

    # Full fine-tuned model
    python src/inference.py --model_path checkpoints/mt5_full_ft/best_model \
        --test_csv data/test.csv --out preds/fullft_preds.json

Output: a JSON file {"preds": [...], "refs": [...]} consumed by evaluate.py.
"""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MAX_INPUT, MAX_TARGET = 512, 84
WS = lambda t: ' '.join(str(t).strip().split())


def load_model(model_path: str, base_model: str, device: str):
    """Load either a LoRA adapter (if adapter_config.json is present) or a full model."""
    is_adapter = os.path.exists(os.path.join(model_path, 'adapter_config.json'))
    if is_adapter:
        from peft import PeftConfig, PeftModel
        pcfg = PeftConfig.from_pretrained(model_path)
        base_name = base_model or pcfg.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name).to(device)
        model = PeftModel.from_pretrained(base, model_path, is_trainable=False).eval()
        print(f"Loaded LoRA adapter from {model_path} (base: {base_name})")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
        print(f"Loaded full model from {model_path}")
    return model, tokenizer


def make_summarizer(model, tokenizer, device: str):
    """Return a function text -> generated headline using the paper's decoding settings."""
    def summarize(text: str) -> str:
        try:
            ids = tokenizer([WS(text)], return_tensors="pt", padding="max_length",
                            truncation=True, max_length=MAX_INPUT)["input_ids"].to(device)
            with torch.no_grad():
                out = model.generate(input_ids=ids, max_length=MAX_TARGET, num_beams=4,
                                     no_repeat_ngram_size=2, length_penalty=0.6)[0]
            return tokenizer.decode(out, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()
        except Exception:
            return str(text)[:100]
    return summarize


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate headlines with a trained model.")
    ap.add_argument('--model_path', required=True, help='Full model dir or LoRA adapter dir.')
    ap.add_argument('--test_csv', required=True, help='Test CSV with columns title, text.')
    ap.add_argument('--out', required=True, help='Output predictions JSON.')
    ap.add_argument('--base_model', default=None,
                    help='Override the base model for a LoRA adapter (optional).')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(args.model_path, args.base_model, device)
    summarize = make_summarizer(model, tokenizer, device)

    test_df = pd.read_csv(args.test_csv)
    preds, refs = [], [WS(t) for t in test_df['title']]
    for i, text in enumerate(test_df['text']):
        preds.append(summarize(text))
        if (i + 1) % 200 == 0:
            print(f"   {i + 1}/{len(test_df)}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'preds': preds, 'refs': refs}, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(preds)} predictions -> {args.out}")


if __name__ == '__main__':
    main()
