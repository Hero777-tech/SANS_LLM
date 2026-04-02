"""
fix_tokenizer.py
================
Run this ONCE to properly convert the raw BPE files
into a working HuggingFace tokenizer.
Place in project root and run: python fix_tokenizer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast

print("=" * 60)
print("  Tokenizer Fix — Rebuilding HF wrapper from BPE files")
print("=" * 60)

VOCAB_FILE  = str(config.TOKENIZER_DIR / "sanskrit-vocab.json")
MERGES_FILE = str(config.TOKENIZER_DIR / "sanskrit-merges.txt")

# ── Step 1: Verify raw files exist and are healthy ────────────
print("\n[1/4] Checking raw BPE files on disk...")
for fpath in [VOCAB_FILE, MERGES_FILE]:
    size_kb = os.path.getsize(fpath) / 1024
    print(f"      {fpath.split(os.sep)[-1]:<30} {size_kb:.1f} KB")
    if size_kb < 10:
        print(f"      ❌ File too small — BPE training may have failed")
        sys.exit(1)
print("      ✅ Raw files look healthy")

# ── Step 2: Reload raw ByteLevelBPE tokenizer from files ──────
print("\n[2/4] Reloading ByteLevelBPETokenizer from raw files...")
bpe_tok = ByteLevelBPETokenizer(
    vocab=VOCAB_FILE,
    merges=MERGES_FILE,
    add_prefix_space=True,
)
print(f"      Raw BPE vocab size: {bpe_tok.get_vocab_size():,}")
if bpe_tok.get_vocab_size() < 15_000:
    print("      ❌ Vocab too small in raw files — need to retrain")
    sys.exit(1)
print("      ✅ Raw BPE tokenizer loaded correctly")

# ── Step 3: Save as unified tokenizer.json ────────────────────
print("\n[3/4] Saving as tokenizer.json (HF unified format)...")
tokenizer_json_path = str(config.TOKENIZER_DIR / "tokenizer.json")
bpe_tok.save(tokenizer_json_path)
size_kb = os.path.getsize(tokenizer_json_path) / 1024
print(f"      tokenizer.json saved — {size_kb:.1f} KB")
print("      ✅ Done")

# ── Step 4: Wrap as PreTrainedTokenizerFast and save ──────────
print("\n[4/4] Wrapping as PreTrainedTokenizerFast...")
hf_tok = PreTrainedTokenizerFast(
    tokenizer_file=tokenizer_json_path,
    bos_token=config.BOS_TOKEN,
    eos_token=config.EOS_TOKEN,
    unk_token=config.UNK_TOKEN,
    pad_token=config.PAD_TOKEN,
    model_max_length=config.MAX_SEQ_LEN,
)

print(f"      Vocab size in HF wrapper: {len(hf_tok):,}")

if len(hf_tok) < 15_000:
    print("      ❌ Still wrong — wrapping failed")
    sys.exit(1)

hf_tok.save_pretrained(str(config.TOKENIZER_DIR))
print("      ✅ Saved with save_pretrained()")

# ── Quick sanity encode test ──────────────────────────────────
print("\n── Quick encode test ──")
test = "नमस्ते। अहं संस्कृतम् अधीयामि।"
ids     = hf_tok.encode(test)
decoded = hf_tok.decode(ids, skip_special_tokens=True)
print(f"   Input   : {test}")
print(f"   N tokens: {len(ids)}")
print(f"   Tokens  : {ids[:12]}...")
print(f"   Decoded : {decoded[:80]}")

print("\n" + "=" * 60)
if len(ids) > 0:
    print("  ✅ FIX SUCCESSFUL — now run: python check_03.py")
else:
    print("  ❌ Encode still returns empty — paste output for help")
print("=" * 60)
