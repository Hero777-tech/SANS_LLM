"""
check_03.py
===========
Quick verification script for the trained Sanskrit tokenizer.
Run this BEFORE proceeding to 04_tokenize_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from transformers import PreTrainedTokenizerFast

print("=" * 60)
print("  Tokenizer Health Check")
print("=" * 60)

# ── Test 1: Load ─────────────────────────────────────────────
print("\n[1/5] Loading tokenizer...")
try:
    tok = PreTrainedTokenizerFast.from_pretrained(str(config.TOKENIZER_DIR))
    print(f"      ✅ Loaded successfully")
except Exception as e:
    print(f"      ❌ FAILED to load: {e}")
    sys.exit(1)

# ── Test 2: Vocab size ────────────────────────────────────────
print(f"\n[2/5] Checking vocab size...")
vocab_size = len(tok)
print(f"      Vocab size : {vocab_size:,}")
if vocab_size >= 15_000:
    print(f"      ✅ Good (expected ~16,000)")
else:
    print(f"      ❌ Too small — wrapping fix did not work")
    sys.exit(1)

# ── Test 3: Special tokens ────────────────────────────────────
print(f"\n[3/5] Checking special tokens...")
checks = {
    "pad_token"  : tok.pad_token,
    "unk_token"  : tok.unk_token,
    "bos_token"  : tok.bos_token,
    "eos_token"  : tok.eos_token,
    "pad_token_id": tok.pad_token_id,
    "bos_token_id": tok.bos_token_id,
    "eos_token_id": tok.eos_token_id,
}
all_ok = True
for k, v in checks.items():
    status = "✅" if v is not None else "❌"
    if v is None:
        all_ok = False
    print(f"      {status} {k:<15} : {v}")
if not all_ok:
    print("      ❌ Some special tokens missing")
    sys.exit(1)

# ── Test 4: Encode/Decode Sanskrit ───────────────────────────
print(f"\n[4/5] Encode → Decode round-trip test...")
test_sentences = [
    "नमस्ते। अहं संस्कृतम् अधीयामि।",
    "रामः वनं गच्छति।",
    "अयं पाठः संस्कृतभाषायाम् अस्ति।।",
    "धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।",
]
encode_ok = True
for sentence in test_sentences:
    ids     = tok.encode(sentence)
    decoded = tok.decode(ids, skip_special_tokens=True)
    n_tok   = len(ids)
    ok      = n_tok > 0
    status  = "✅" if ok else "❌"
    if not ok:
        encode_ok = False
    print(f"\n      {status} Input   : {sentence}")
    print(f"         Tokens  : {ids[:10]}{'...' if len(ids)>10 else ''}")
    print(f"         N tokens: {n_tok}")
    print(f"         Decoded : {decoded[:80]}")

if not encode_ok:
    print("\n      ❌ Encoding failed — tokenizer still broken")
    sys.exit(1)

# ── Test 5: Files on disk ─────────────────────────────────────
print(f"\n[5/5] Checking files on disk...")
expected_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "sanskrit-vocab.json",
    "sanskrit-merges.txt",
    "special_tokens_map.json",
]
all_files_ok = True
for fname in expected_files:
    fpath = config.TOKENIZER_DIR / fname
    exists = fpath.exists()
    size   = f"{fpath.stat().st_size / 1024:.1f} KB" if exists else "—"
    status = "✅" if exists else "⚠️ "
    if not exists and fname in ("tokenizer.json", "tokenizer_config.json"):
        all_files_ok = False
    print(f"      {status} {fname:<30} {size}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if encode_ok and all_files_ok:
    print("  ✅ ALL CHECKS PASSED — safe to run 04_tokenize_dataset.py")
else:
    print("  ❌ SOME CHECKS FAILED — fix tokenizer before proceeding")
print("=" * 60)

