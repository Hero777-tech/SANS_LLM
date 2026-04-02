"""
04_tokenize_dataset.py
======================
STEP 4 — Memory-safe version for 16GB RAM.
Tokenizes corpus in streaming chunks, writes directly to disk.
Never loads more than STREAM_CHUNK lines into RAM at once.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logging_utils import log, log_section, log_stats, Timer

import gc
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm


# ── Memory-safe config ────────────────────────────────────────
STREAM_CHUNK    = 500       # Lines read from corpus at a time (keep LOW)
ENCODE_BATCH    = 200       # Lines encoded per tokenizer call
CHUNK_FLUSH     = 5_000     # Sequences accumulated before writing to disk
SEQ_LEN         = config.MAX_SEQ_LEN   # 512


def load_tokenizer():
    if not config.TOKENIZER_DIR.exists():
        log("Tokenizer not found. Run fix_tokenizer.py first.", "❌")
        sys.exit(1)
    tok = PreTrainedTokenizerFast.from_pretrained(str(config.TOKENIZER_DIR))
    log(f"Tokenizer loaded. Vocab: {len(tok):,}", "✅")
    return tok


def count_lines(filepath) -> int:
    log("Counting corpus lines...", "🔢")
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    log(f"Total lines: {count:,}", "📊")
    return count


def iter_corpus_chunks(filepath, chunk_size):
    """Yield small chunks of non-empty lines — never loads full file."""
    chunk = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk.clear()
    if chunk:
        yield chunk


def encode_chunk(tokenizer, lines, bos_id, eos_id) -> list[int]:
    """Encode a small batch of lines → flat token id list."""
    flat = []
    encoded = tokenizer(
        lines,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    for ids in encoded["input_ids"]:
        flat.append(bos_id)
        flat.extend(ids)
        flat.append(eos_id)
    return flat


def flush_sequences_to_disk(sequences: list, out_dir: Path, shard_idx: int) -> Path:
    """Save a list of sequence dicts as an Arrow shard file."""
    shard_path = out_dir / f"shard_{shard_idx:04d}"
    ds = HFDataset.from_list(sequences)
    ds.save_to_disk(str(shard_path))
    return shard_path


def merge_shards(shard_paths: list, out_path: Path):
    """Merge all shards into a single dataset."""
    from datasets import concatenate_datasets, load_from_disk
    log(f"Merging {len(shard_paths)} shards...", "🔗")
    datasets = []
    for sp in tqdm(shard_paths, desc="Loading shards"):
        datasets.append(load_from_disk(str(sp)))
    merged = concatenate_datasets(datasets)
    merged.save_to_disk(str(out_path))
    log(f"Merged dataset: {len(merged):,} sequences", "✅")
    return merged


def cleanup_shards(shard_paths: list):
    """Delete temporary shard folders."""
    import shutil
    for sp in shard_paths:
        shutil.rmtree(sp, ignore_errors=True)
    log("Temporary shards deleted", "🗑️")


def main():
    log_section("STEP 4 — Memory-Safe Dataset Tokenization")

    if not config.CORPUS_FILE.exists():
        log("Corpus not found. Run 02_preprocess.py first.", "❌")
        sys.exit(1)

    # Check if already done
    train_path = config.TOKENIZED_DIR / "train"
    eval_path  = config.TOKENIZED_DIR / "eval"
    if train_path.exists() and eval_path.exists():
        log("Tokenized dataset already exists.", "✅")
        if input("Re-tokenize? [y/N]: ").strip().lower() != "y":
            log("Skipping. Next step: python 05_train.py", "👉")
            return

    config.TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    shards_dir = config.TOKENIZED_DIR / "_shards"
    shards_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    bos_id    = tokenizer.bos_token_id
    eos_id    = tokenizer.eos_token_id

    total_lines = count_lines(config.CORPUS_FILE)

    log_stats("Tokenization Config", {
        "corpus_file"   : str(config.CORPUS_FILE),
        "stream_chunk"  : STREAM_CHUNK,
        "encode_batch"  : ENCODE_BATCH,
        "seq_len"       : SEQ_LEN,
        "flush_every"   : CHUNK_FLUSH,
        "ram_target"    : "< 4GB active",
    })

    # ── Main tokenization loop ────────────────────────────────
    token_buffer    = []    # Rolling token id buffer
    seq_buffer      = []    # Accumulated sequences before flush
    shard_paths     = []
    shard_idx       = 0
    total_seqs      = 0
    total_tokens    = 0
    lines_done      = 0

    pbar = tqdm(total=total_lines, desc="Tokenizing", unit="doc")

    with Timer("Full tokenization"):
        for chunk in iter_corpus_chunks(config.CORPUS_FILE, STREAM_CHUNK):

            # Encode in sub-batches
            for i in range(0, len(chunk), ENCODE_BATCH):
                sub = chunk[i : i + ENCODE_BATCH]
                ids = encode_chunk(tokenizer, sub, bos_id, eos_id)
                token_buffer.extend(ids)
                total_tokens += len(ids)

            lines_done += len(chunk)
            pbar.update(len(chunk))

            # Slice token_buffer into SEQ_LEN sequences
            while len(token_buffer) >= SEQ_LEN:
                seq = token_buffer[:SEQ_LEN]
                seq_buffer.append({"input_ids": seq})
                token_buffer = token_buffer[SEQ_LEN:]
                total_seqs += 1

            # Flush to disk when buffer is large enough
            if len(seq_buffer) >= CHUNK_FLUSH:
                shard_path = flush_sequences_to_disk(
                    seq_buffer, shards_dir, shard_idx
                )
                shard_paths.append(shard_path)
                shard_idx += 1
                seq_buffer.clear()
                gc.collect()    # Force RAM release

                pbar.set_postfix({
                    "seqs"  : f"{total_seqs:,}",
                    "shards": shard_idx,
                    "tokM"  : f"{total_tokens/1e6:.0f}M",
                })

        # Final flush
        if seq_buffer:
            shard_path = flush_sequences_to_disk(
                seq_buffer, shards_dir, shard_idx
            )
            shard_paths.append(shard_path)
            seq_buffer.clear()

        # Remaining tokens that don't fill a full sequence — discard
        leftover = len(token_buffer)
        token_buffer.clear()
        gc.collect()

    pbar.close()

    log_stats("Tokenization Complete", {
        "total_lines"       : total_lines,
        "total_tokens"      : total_tokens,
        "total_sequences"   : total_seqs,
        "leftover_tokens"   : leftover,
        "shards_written"    : len(shard_paths),
        "seq_length"        : SEQ_LEN,
    })

    # ── Merge shards ──────────────────────────────────────────
    log_section("Merging Shards")

    with Timer("Merging"):
        all_dataset = merge_shards(shard_paths, config.TOKENIZED_DIR / "_merged")

    gc.collect()

    # ── Train / eval split ────────────────────────────────────
    log("Splitting train / eval (99% / 1%)...", "✂️")
    n       = len(all_dataset)
    n_eval  = max(100, int(n * 0.01))
    n_train = n - n_eval

    train_ds = all_dataset.select(range(n_train))
    eval_ds  = all_dataset.select(range(n_train, n))

    log_stats("Split", {
        "train_sequences": len(train_ds),
        "eval_sequences" : len(eval_ds),
    })

    # ── Save final datasets ───────────────────────────────────
    log(f"Saving train → {train_path}", "💾")
    with Timer("Save train"):
        train_ds.save_to_disk(str(train_path))

    log(f"Saving eval  → {eval_path}", "💾")
    with Timer("Save eval"):
        eval_ds.save_to_disk(str(eval_path))

    # ── Cleanup ───────────────────────────────────────────────
    import shutil
    shutil.rmtree(config.TOKENIZED_DIR / "_merged", ignore_errors=True)
    cleanup_shards(shard_paths)
    gc.collect()

    # ── Final summary ─────────────────────────────────────────
    train_mb = sum(
        f.stat().st_size for f in train_path.rglob("*") if f.is_file()
    ) / 1e6

    log_stats("Final Dataset", {
        "train_sequences"   : len(train_ds),
        "eval_sequences"    : len(eval_ds),
        "train_size_MB"     : round(train_mb, 1),
        "seq_length"        : SEQ_LEN,
    })

    log("✅ Step 4 complete!", "🎉")
    log("Next step: python 05_train.py", "👉")


if __name__ == "__main__":
    main()