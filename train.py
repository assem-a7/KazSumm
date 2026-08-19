"""
train.py
========
Fine-tune mT5-base on KazSumm, either with full fine-tuning or with LoRA
(parameter-efficient), reproducing the exact configuration used in the paper.

Shared configuration
--------------------
    base model            : csebuetnlp/mT5_multilingual_XLSum
    max input / target    : 512 / 84 tokens
    effective batch size   : 4 x 4 (per-device batch x gradient accumulation)
    epochs                : 3   (early stopping, patience = 2, on eval_loss)
    warmup steps          : 200
    weight decay          : 0.01
    optimizer             : Adafactor
    lr scheduler          : linear
    precision             : bf16 (mT5 is unstable in fp16), else fp32
    seed                  : 42

Mode-specific
-------------
    full : learning rate 5e-4, all 582M parameters trained
    lora : learning rate 3e-4, r in {8, 16}, alpha 32, dropout 0.1,
           target modules q, v  (~0.15% / 0.30% of parameters trained)

Usage
-----
    python src/train.py --mode full  --data_dir data --out checkpoints/mt5_full_ft
    python src/train.py --mode lora --lora_r 8  --data_dir data --out checkpoints/mt5_lora_r8
    python src/train.py --mode lora --lora_r 16 --data_dir data --out checkpoints/mt5_lora_r16

The best checkpoint (lowest eval_loss) is saved to <out>/best_model.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

SEED = 42
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
MAX_INPUT, MAX_TARGET = 512, 84
WS = lambda t: ' '.join(str(t).strip().split())


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess(examples, tokenizer):
    """Tokenise article bodies (inputs) and headlines (labels)."""
    model_inputs = tokenizer([WS(t) for t in examples['text']],
                             max_length=MAX_INPUT, truncation=True, padding=False)
    labels = tokenizer(text_target=[WS(t) for t in examples['title']],
                       max_length=MAX_TARGET, truncation=True, padding=False)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def build_model(mode: str, lora_r: int, base_model: str, device: str):
    """Load mT5-base and, for LoRA mode, wrap it with a LoRA adapter."""
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if mode == 'lora':
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=lora_r, lora_alpha=32,
                         lora_dropout=0.1, target_modules=["q", "v"], bias="none")
        model = get_peft_model(base, cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA r={lora_r}: {trainable:,} trainable params "
              f"({trainable / total * 100:.2f}%) on modules q, v")
    else:
        model = base
        print("Full fine-tuning: all parameters trainable")
    return model.to(device)


def train(args) -> None:
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_bf16 = device == 'cuda' and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    train_df = pd.read_csv(f'{args.data_dir}/train.csv')
    val_df = pd.read_csv(f'{args.data_dir}/val.csv')

    tr_tok = Dataset.from_pandas(train_df[['title', 'text']]).map(
        lambda e: preprocess(e, tokenizer), batched=True,
        remove_columns=['text', 'title'], desc='tok/train')
    va_tok = Dataset.from_pandas(val_df[['title', 'text']]).map(
        lambda e: preprocess(e, tokenizer), batched=True,
        remove_columns=['text', 'title'], desc='tok/val')

    lr = args.lr if args.lr is not None else (3e-4 if args.mode == 'lora' else 5e-4)
    model = build_model(args.mode, args.lora_r, args.base_model, device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                      padding=True, pad_to_multiple_of=8)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,   # effective batch = batch * grad_accum
        learning_rate=lr,
        warmup_steps=200,
        weight_decay=0.01,
        optim="adafactor",
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=False,
        bf16=use_bf16,
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=tr_tok,
        eval_dataset=va_tok,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"Training ({args.mode}, lr={lr}, bf16={use_bf16}) ...")
    trainer.train()

    best_dir = f'{args.out}/best_model'
    os.makedirs(best_dir, exist_ok=True)
    model.save_pretrained(best_dir)        # LoRA: saves the adapter; full: saves the model
    tokenizer.save_pretrained(best_dir)
    print(f"Saved best model to {best_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune mT5 on KazSumm (full or LoRA).")
    ap.add_argument('--mode', choices=['full', 'lora'], required=True)
    ap.add_argument('--lora_r', type=int, default=16, help='LoRA rank (used only in lora mode).')
    ap.add_argument('--data_dir', default='data', help='Directory with train.csv and val.csv.')
    ap.add_argument('--out', required=True, help='Output directory for checkpoints.')
    ap.add_argument('--base_model', default=MODEL_NAME)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=None,
                    help='Override learning rate (default: 3e-4 lora / 5e-4 full).')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
