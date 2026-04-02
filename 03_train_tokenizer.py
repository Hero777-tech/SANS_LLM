"""
03_train_tokenizer.py
=====================
STEP 3 — Train a Byte-Level BPE tokenizer on the Sanskrit corpus.
Output: data/tokenizer/  (vocab, merges, HF tokenizer)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logging_utils import log, log_section, log_stats, Timer
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


def corpus_line_iterator(corpus_file, batch_size=1_000):
    batch = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)
                if len(batch) >= batch_size:
                    yield from batch
                    batch.clear()
    if batch:
        yield from batch


def count_corpus_lines(corpus_file) -> int:
    count = 0
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    log_section("STEP 3 — Tokenizer Training")

    if not config.CORPUS_FILE.exists():
        log("Corpus not found. Run 02_preprocess.py first.", "❌")
        sys.exit(1)

    if config.VOCAB_FILE.exists() and config.MERGES_FILE.exists():
        log("Tokenizer already exists.", "✅")
        if input("Re-train? [y/N]: ").strip().lower() != "y":
            _verify_tokenizer()
            log("Next step: python 04_tokenize_dataset.py", "👉")
            return

    config.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    log("Counting corpus lines...", "🔢")
    n_lines = count_corpus_lines(config.CORPUS_FILE)
    log(f"Total documents: {n_lines:,}", "📊")

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)

    log_stats("Tokenizer Training Config", {
        "vocab_size"    : config.VOCAB_SIZE,
        "min_frequency" : config.MIN_FREQUENCY,
        "special_tokens": config.SPECIAL_TOKENS,
    })
    log("Training tokenizer (5–15 minutes for large corpus)...", "⏳")

    with Timer("Tokenizer training"):
        tokenizer.train_from_iterator(
            iterator=corpus_line_iterator(config.CORPUS_FILE, config.TOKENIZER_BATCH),
            vocab_size=config.VOCAB_SIZE,
            min_frequency=config.MIN_FREQUENCY,
            special_tokens=config.SPECIAL_TOKENS,
            show_progress=True,
            length=n_lines,
        )

    tokenizer.save_model(str(config.TOKENIZER_DIR), "sanskrit")

    hf_tokenizer = GPT2TokenizerFast(
        vocab_file=str(config.VOCAB_FILE),
        merges_file=str(config.MERGES_FILE),
        unk_token=config.UNK_TOKEN,
        bos_token=config.BOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        pad_token=config.PAD_TOKEN,
        add_prefix_space=True,
    )
    hf_tokenizer.save_pretrained(str(config.TOKENIZER_DIR))
    log(f"Tokenizer saved. Vocab: {len(hf_tokenizer):,}", "💾")

    _verify_tokenizer(hf_tokenizer)
    log("Next step: run  python 04_tokenize_dataset.py", "👉")


def _verify_tokenizer(tokenizer=None):
    log_section("Tokenizer Verification")
    if tokenizer is None:
        tokenizer = GPT2TokenizerFast.from_pretrained(str(config.TOKENIZER_DIR))

    for sentence in [
        "नमस्ते। अहं संस्कृतम् अधीयामि।",
        "रामः वनं गच्छति।",
    ]:
        enc = tokenizer.encode(sentence)
        dec = tokenizer.decode(enc)
        print(f"\n  Input   : {sentence}")
        print(f"  Tokens  : {enc[:15]}...")
        print(f"  N tokens: {len(enc)}")
        print(f"  Decoded : {dec}")


if __name__ == "__main__":
    main()