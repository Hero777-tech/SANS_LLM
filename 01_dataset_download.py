"""
01_dataset_download.py
======================
STEP 1 — Download ai4bharat/sangraha Sanskrit dataset and save to disk.
Run once. Output: data/raw/  (HuggingFace Arrow format)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logging_utils import log, log_section, log_stats, Timer
from datasets import load_dataset


def main():
    log_section("STEP 1 — Dataset Download")
    config.print_config()

    if config.RAW_DIR.exists() and any(config.RAW_DIR.iterdir()):
        log(f"Raw dataset already exists at: {config.RAW_DIR}", "✅")
        log("Delete that folder and re-run if you want a fresh download.", "ℹ️")
        return

    log(f"Downloading: {config.DATASET_NAME} ({config.DATASET_SUBSET})", "📥")
    log("This may take a while on first run (several GB download)...", "⏳")

    with Timer("Dataset download"):
        raw_dataset = load_dataset(
            config.DATASET_NAME,
            data_dir=config.DATASET_SUBSET,
            split=config.DATASET_SPLIT,
            trust_remote_code=True,
        )

    text_col = config.TEXT_COLUMN if config.TEXT_COLUMN in raw_dataset.column_names \
               else raw_dataset.column_names[0]

    log_stats("Raw Dataset Info", {
        "total_samples" : len(raw_dataset),
        "columns"       : str(raw_dataset.column_names),
        "text_column"   : text_col,
    })

    for i in range(min(3, len(raw_dataset))):
        sample_text = str(raw_dataset[i][text_col])
        print(f"\n  ── Sample {i+1} ──")
        print(f"  {sample_text[:200]}")

    # Estimate size
    sample_size = sum(len(str(raw_dataset[i][text_col])) for i in range(min(1000, len(raw_dataset))))
    avg_chars = sample_size / min(1000, len(raw_dataset))
    total_estimated = avg_chars * len(raw_dataset)

    log_stats("Size Estimate", {
        "avg_chars_per_doc"     : int(avg_chars),
        "estimated_total_MB"    : round(total_estimated / 1e6, 1),
        "estimated_tokens"      : int(total_estimated / 4),
    })

    log(f"Saving dataset to: {config.RAW_DIR}", "💾")
    with Timer("Saving to disk"):
        raw_dataset.save_to_disk(str(config.RAW_DIR))

    log(f"✅ Dataset saved to {config.RAW_DIR}", "🎉")
    log("Next step: run  python 02_preprocess.py", "👉")


if __name__ == "__main__":
    main()