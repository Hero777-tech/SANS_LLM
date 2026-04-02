"""
05_train.py
===========
STEP 5 — Train GPT-2 Small from scratch on tokenized Sanskrit dataset.

Usage:
    python 05_train.py               # Full training
    python 05_train.py --smoke-test  # Quick 100-step sanity check
"""

import sys, os, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logging_utils import log, log_section, log_stats, Timer
import torch
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
                           DataCollatorForLanguageModeling, Trainer,
                           TrainingArguments, set_seed)
from datasets import load_from_disk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        log_stats("GPU", {"name": props.name, "VRAM_GB": round(props.total_memory/1e9,2)})
        if props.total_memory < 7e9:
            log("⚠️  <8GB VRAM — reduce batch size in config.py!", "⚠️")
    else:
        log("No GPU found — CPU training will be very slow!", "⚠️")
    return device


def load_tokenizer_and_datasets():
    if not config.TOKENIZER_DIR.exists():
        log("Tokenizer not found.", "❌"); sys.exit(1)
    tokenizer = GPT2TokenizerFast.from_pretrained(str(config.TOKENIZER_DIR))
    train_path, eval_path = config.TOKENIZED_DIR/"train", config.TOKENIZED_DIR/"eval"
    if not train_path.exists():
        log("Tokenized dataset not found.", "❌"); sys.exit(1)
    train_ds = load_from_disk(str(train_path))
    eval_ds  = load_from_disk(str(eval_path))
    log_stats("Datasets", {"train": len(train_ds), "eval": len(eval_ds)})
    return tokenizer, train_ds, eval_ds


def build_model(tokenizer):
    cfg = config.MODEL_CONFIG.copy()
    cfg["vocab_size"] = len(tokenizer)
    model = GPT2LMHeadModel(GPT2Config(**cfg))
    if config.TRAINING_CONFIG.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        log("Gradient checkpointing enabled", "✅")
    total = sum(p.numel() for p in model.parameters())
    log_stats("Model (GPT-2 Small)", {
        "parameters": f"{total/1e6:.1f}M",
        "layers": cfg["n_layer"], "heads": cfg["n_head"],
        "context": cfg["n_positions"], "vocab": cfg["vocab_size"],
    })
    return model


def find_latest_checkpoint():
    ckpts = [d for d in config.CHECKPOINT_DIR.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: int(p.name.split("-")[-1]))
    log(f"Found checkpoint: {ckpts[-1]}", "🔄")
    return str(ckpts[-1])


def main():
    args = parse_args()
    set_seed(config.TRAINING_CONFIG["seed"])
    log_section("STEP 5 — GPT-2 Training")
    config.print_config()

    check_gpu()
    tokenizer, train_ds, eval_ds = load_tokenizer_and_datasets()

    if args.smoke_test:
        log("🔬 SMOKE TEST — 100 steps only", "⚠️")
        train_ds = train_ds.select(range(min(512, len(train_ds))))
        eval_ds  = eval_ds.select(range(min(64, len(eval_ds))))

    model = build_model(tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    train_args_dict = config.TRAINING_CONFIG.copy()
    if args.smoke_test:
        train_args_dict.update({"max_steps":100,"eval_steps":50,"save_steps":50,"logging_steps":10})

    training_args = TrainingArguments(**train_args_dict)

    resume = args.resume or find_latest_checkpoint()
    if not resume:
        log("Starting fresh training run", "🆕")

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    )

    log_section("Starting Training")
    log("Monitor: tensorboard --logdir " + str(config.LOGS_DIR), "📈")

    with Timer("Training"):
        result = trainer.train(resume_from_checkpoint=resume)

    final = config.CHECKPOINT_DIR / "final_model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))

    eval_results = trainer.evaluate()
    log_stats("Results", {
        "train_loss": round(result.training_loss, 4),
        "eval_loss" : round(eval_results["eval_loss"], 4),
        "perplexity": round(math.exp(eval_results["eval_loss"]), 2),
        "steps"     : result.global_step,
    })

    log(f"✅ Done! Model saved: {final}", "🎉")
    log("Next step: run  python 06_generate.py", "👉")


if __name__ == "__main__":
    main()
    
    
    
    