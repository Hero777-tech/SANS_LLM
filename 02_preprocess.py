"""
02_preprocess.py
================
STEP 2 — Clean and filter the raw dataset.
Output: data/corpus/sanskrit_corpus.txt
Each line = one cleaned Sanskrit document.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logging_utils import log, log_section, log_stats, Timer
from utils.text_cleaner import clean_and_filter, text_stats
from datasets import load_from_disk
from tqdm.auto import tqdm

PROCESS_BATCH = 5_000
WRITE_BATCH   = 500


def detect_text_column(dataset) -> str:
    if config.TEXT_COLUMN in dataset.column_names:
        return config.TEXT_COLUMN
    col = dataset.column_names[0]
    log(f"text column not found, using: '{col}'", "⚠️")
    return col


def process_batch(batch_items: list, text_col: str) -> tuple[list, int]:
    kept, skipped = [], 0
    for item in batch_items:
        result = clean_and_filter(
            str(item[text_col]),
            min_words=config.CLEANING_CONFIG["min_words"],
            min_devanagari_ratio=config.CLEANING_CONFIG["min_devanagari_ratio"],
            max_chars=config.CLEANING_CONFIG["max_doc_chars"],
        )
        if result is not None:
            kept.append(result)
        else:
            skipped += 1
    return kept, skipped


def main():
    log_section("STEP 2 — Preprocessing & Corpus Building")

    if not config.RAW_DIR.exists():
        log("Raw dataset not found. Run 01_dataset_download.py first.", "❌")
        sys.exit(1)

    log(f"Loading raw dataset from: {config.RAW_DIR}", "📂")
    with Timer("Load dataset"):
        dataset = load_from_disk(str(config.RAW_DIR))

    text_col = detect_text_column(dataset)
    total = len(dataset)

    if config.MAX_SAMPLES is not None:
        total = min(config.MAX_SAMPLES, total)
        dataset = dataset.select(range(total))
        log(f"Limited to {total:,} samples", "✂️")
    else:
        log(f"Processing full dataset: {total:,} samples", "📊")

    if config.CORPUS_FILE.exists():
        size_mb = config.CORPUS_FILE.stat().st_size / 1e6
        log(f"Corpus already exists ({size_mb:.1f} MB)", "✅")
        if input("Re-process? [y/N]: ").strip().lower() != "y":
            log("Next step: python 03_train_tokenizer.py", "👉")
            return

    total_kept = total_skipped = total_chars = total_words = 0
    buffer = []
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    with Timer("Full preprocessing"):
        with open(config.CORPUS_FILE, "w", encoding="utf-8") as f_out:
            pbar = tqdm(total=total, desc="Cleaning documents", unit="doc")

            for batch_start in range(0, total, PROCESS_BATCH):
                batch_end = min(batch_start + PROCESS_BATCH, total)
                batch = [dataset[i] for i in range(batch_start, batch_end)]
                kept, skipped = process_batch(batch, text_col)
                total_kept += len(kept)
                total_skipped += skipped

                for text in kept:
                    stats = text_stats(text)
                    total_chars += stats["char_count"]
                    total_words += stats["word_count"]
                    buffer.append(text)

                if len(buffer) >= WRITE_BATCH:
                    f_out.write("\n".join(buffer) + "\n")
                    f_out.flush()
                    buffer.clear()

                pbar.update(batch_end - batch_start)
                pbar.set_postfix({"kept": f"{total_kept:,}", "chars": f"{total_chars/1e6:.1f}M"})

            if buffer:
                f_out.write("\n".join(buffer) + "\n")
            pbar.close()

    log_stats("Preprocessing Complete", {
        "kept_docs"       : total_kept,
        "skipped_docs"    : total_skipped,
        "keep_rate_%"     : round(100 * total_kept / max(total, 1), 1),
        "total_chars"     : total_chars,
        "total_words"     : total_words,
        "estimated_tokens": total_words * 4,
        "corpus_MB"       : round(config.CORPUS_FILE.stat().st_size / 1e6, 2),
    })

    log("Next step: run  python 03_train_tokenizer.py", "👉")


if __name__ == "__main__":
    main()