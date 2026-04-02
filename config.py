"""
config.py
=========
Central configuration file for the Sanskrit GPT-2 LLM project.
Edit this file to change paths, model size, or training hyperparameters.
All other scripts import from here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# 📁 PROJECT PATHS
# ─────────────────────────────────────────────

# Base project directory — change this to your actual project root
BASE_DIR = Path(r"M:/research papers/sanskrit paper work/SANSKRIT_LLM_GPT2")

# Data directories (auto-created by scripts)
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"           # Downloaded HF dataset (Arrow)
CORPUS_DIR      = DATA_DIR / "corpus"        # Plain text corpus
TOKENIZER_DIR   = DATA_DIR / "tokenizer"     # Trained BPE tokenizer
TOKENIZED_DIR   = DATA_DIR / "tokenized"     # Tokenized HF dataset

# Output directories
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"   # Training checkpoints
LOGS_DIR        = BASE_DIR / "logs"          # TensorBoard logs

# File paths
CORPUS_FILE     = CORPUS_DIR / "sanskrit_corpus.txt"
VOCAB_FILE      = TOKENIZER_DIR / "sanskrit-vocab.json"
MERGES_FILE     = TOKENIZER_DIR / "sanskrit-merges.txt"

# ─────────────────────────────────────────────
# 📥 DATASET CONFIG
# ─────────────────────────────────────────────

DATASET_NAME    = "ai4bharat/sangraha"
DATASET_SUBSET  = "verified/san"
DATASET_SPLIT   = "train"
TEXT_COLUMN     = "text"

# Set to None to use full dataset
MAX_SAMPLES     = None

# ─────────────────────────────────────────────
# 🔤 TOKENIZER CONFIG
# ─────────────────────────────────────────────

VOCAB_SIZE          = 16_000
MIN_FREQUENCY       = 2
TOKENIZER_BATCH     = 1_000

# Special tokens
PAD_TOKEN   = "<pad>"
UNK_TOKEN   = "<unk>"
BOS_TOKEN   = "<s>"
EOS_TOKEN   = "</s>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

# ─────────────────────────────────────────────
# 🧠 MODEL CONFIG (GPT-2 Small)
# ─────────────────────────────────────────────

MODEL_CONFIG = {
    "vocab_size"        : VOCAB_SIZE,
    "n_positions"       : 512,
    "n_ctx"             : 512,
    "n_embd"            : 768,
    "n_layer"           : 12,
    "n_head"            : 12,
    "activation_function": "gelu_new",
    "resid_pdrop"       : 0.1,
    "embd_pdrop"        : 0.1,
    "attn_pdrop"        : 0.1,
    "bos_token_id"      : 2,
    "eos_token_id"      : 3,
}

MAX_SEQ_LEN = MODEL_CONFIG["n_positions"]

# ─────────────────────────────────────────────
# ⚙️ TOKENIZATION CONFIG
# ─────────────────────────────────────────────

TOKENIZE_BATCH_SIZE = 1_000
NUM_PROC            = 4

# ─────────────────────────────────────────────
# 🏋️ TRAINING CONFIG  (RTX 2060 Super tuned)
# ─────────────────────────────────────────────

TRAINING_CONFIG = {
    "output_dir"                        : str(CHECKPOINT_DIR),
    "per_device_train_batch_size"       : 8, #4 only
    "gradient_accumulation_steps"       : 8,
    "num_train_epochs"                  : 3,
    "max_steps"                         : -1,
    "learning_rate"                     : 3e-4,
    "weight_decay"                      : 0.01,
    "warmup_steps"                      : 5000,
    "lr_scheduler_type"                 : "cosine",
    "adam_beta1"                        : 0.9,
    "adam_beta2"                        : 0.999,
    "adam_epsilon"                      : 1e-8,
    "max_grad_norm"                     : 1.0,
    "fp16"                              : True,
    "gradient_checkpointing"            : False, #True
    "logging_steps"                     : 100,
    "save_steps"                        : 1_000,
    "save_total_limit"                  : 3,
    "eval_steps"                        : 1_000,
    "eval_strategy"                     : "steps",
    "dataloader_num_workers"            : 8, #2
    "seed"                              : 42,
    "report_to"                         : "none",
    "load_best_model_at_end"            : True,
    "metric_for_best_model"             : "eval_loss",
    "greater_is_better"                 : False,
    # "loss_type"                         : "ForCausalLMLoss",
}

# ─────────────────────────────────────────────
# 📝 TEXT CLEANING CONFIG
# ─────────────────────────────────────────────

CLEANING_CONFIG = {
    "min_words"             : 10,
    "min_devanagari_ratio"  : 0.6,
    "max_doc_chars"         : 50_000,
}

# ─────────────────────────────────────────────
# 🔧 ENVIRONMENT
# ─────────────────────────────────────────────

HF_HOME = str(BASE_DIR)
os.environ["HF_HOME"] = HF_HOME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for _dir in [DATA_DIR, RAW_DIR, CORPUS_DIR, TOKENIZER_DIR, TOKENIZED_DIR,
             CHECKPOINT_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


def print_config():
    print("=" * 60)
    print("📋 Sanskrit LLM Configuration")
    print("=" * 60)
    print(f"  Base dir       : {BASE_DIR}")
    print(f"  Dataset        : {DATASET_NAME} ({DATASET_SUBSET})")
    print(f"  Max samples    : {MAX_SAMPLES or 'ALL'}")
    print(f"  Vocab size     : {VOCAB_SIZE:,}")
    print(f"  Context length : {MAX_SEQ_LEN}")
    print(f"  Model params   : ~124M (GPT-2 Small)")
    print(f"  Batch size     : {TRAINING_CONFIG['per_device_train_batch_size']} "
          f"× {TRAINING_CONFIG['gradient_accumulation_steps']} steps "
          f"= {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']} effective")
    print(f"  FP16           : {TRAINING_CONFIG['fp16']}")
    print(f"  Epochs         : {TRAINING_CONFIG['num_train_epochs']}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()