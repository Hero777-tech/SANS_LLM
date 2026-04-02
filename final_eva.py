"""
08_full_eval.py
===============
Comprehensive Sanskrit LLM Evaluation Suite
Covers: Perplexity, Repetition, Relevance, Script Quality,
        Devanagari Purity, Lexical Diversity, Sentence Completion,
        OOD Robustness, Length Adequacy, Avg Token Entropy

Usage:
    python 08_full_eval.py
"""

import sys, os, math, re, json
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ──────────────────────────────────────────────
#  TEST SUITE  (edit prompts here)
# ──────────────────────────────────────────────

TEST_SUITE = {

    "philosophical": [
        "ब्रह्म सत्यम् जगत् मिथ्या",
        "आत्मा नित्यः शाश्वतः",
        "तत्त्वमसि",
        "अहं ब्रह्मास्मि",
    ],

    "enumeration": [
        "पञ्चभूतानि",
        "चतुर्वर्णाः",
        "अष्टाङ्गयोगः",
        "षट्कर्माणि",
    ],

    "poetic_instruction": [
        "एकं सुन्दरं प्रकृतिवर्णनं रचय।",
        "पर्वतानां नदीनां च वर्णनं कुरु।",
        "धर्मविषयकं श्लोकं लिख।",
        "महाभारतात् एकं श्लोकं उद्धर।",
    ],

    "chapter_reference": [
        "द्वितीयोऽध्यायः",
        "गीता अध्याय २ श्लोक १",
        "प्रथमः पाठः",
    ],

    "ood_stress": [
        "hello world",
        "my name is",
        "१ + १ =",
        "२ + २ =",
        "३ × ३ =",
    ],
}

# ──────────────────────────────────────────────
#  GENERATION CONFIG
# ──────────────────────────────────────────────

GEN_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.92,
    "do_sample": True,
}

# ──────────────────────────────────────────────
#  CHECKPOINT LOADER
# ──────────────────────────────────────────────

def find_best_checkpoint():
    final = config.CHECKPOINT_DIR / "final_model"
    if final.exists():
        return str(final)
    ckpts = [d for d in config.CHECKPOINT_DIR.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")]
    if ckpts:
        ckpts.sort(key=lambda p: int(p.name.split("-")[-1]))
        return str(ckpts[-1])
    return str(config.CHECKPOINT_DIR)


def load_model():
    checkpoint = find_best_checkpoint()
    print(f"📂 Loading checkpoint: {checkpoint}")
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint)
    model     = GPT2LMHeadModel.from_pretrained(checkpoint)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device).eval()
    print(f"✅ Model loaded on {device}\n")
    return model, tokenizer, device

# ──────────────────────────────────────────────
#  GENERATION
# ──────────────────────────────────────────────

def generate(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **GEN_CONFIG,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    gen  = full[len(prompt):].strip()
    return gen

# ──────────────────────────────────────────────
#  METRIC FUNCTIONS
# ──────────────────────────────────────────────

# 1. Perplexity (lower = better)
def metric_perplexity(model, tokenizer, device, text):
    if not text.strip():
        return 9999.0
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return round(math.exp(out.loss.item()), 2)


# 2. Repetition ratio — unique tokens / total tokens (higher = better)
def metric_repetition(text):
    words = text.split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


# 3. Lexical diversity — type-token ratio on characters (for agglutinative Sanskrit)
def metric_ttr(text):
    chars = list(text.replace(" ", ""))
    if not chars:
        return 0.0
    bigrams = [chars[i] + chars[i+1] for i in range(len(chars)-1)]
    return round(len(set(bigrams)) / len(bigrams), 3) if bigrams else 0.0


# 4. Devanagari script purity — % of chars that are Devanagari / punctuation
def metric_devanagari_purity(text):
    if not text:
        return 0.0
    total = len(text.replace(" ", ""))
    if total == 0:
        return 0.0
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F'
               or c in '।॥०१२३४५६७८९ ़')
    return round(deva / total, 3)


# 5. Sentence completion — does output contain proper Sanskrit sentence endings?
def metric_sentence_completion(text):
    return 1.0 if ('।' in text or '॥' in text) else 0.0


# 6. Length adequacy — normalised token count (target: 40-120 tokens)
def metric_length(text):
    n = len(text.split())
    if n == 0:
        return 0.0
    if n >= 40:
        return 1.0
    return round(n / 40, 2)


# 7. Prompt absorption — does the output begin with a natural Sanskrit continuation?
def metric_prompt_absorption(prompt, output):
    # Check if output starts mid-word (bad) vs complete token (good)
    if not output:
        return 0.0
    first_char = output[0]
    # Mid-vowel diacritic as first char = absorbed incorrectly
    bad_starts = set('ािीुूृेैोौंःँ')
    if first_char in bad_starts:
        return 0.3
    # Starts with Devanagari consonant or full word = good
    if '\u0900' <= first_char <= '\u097F':
        return 1.0
    return 0.5


# 8. OOD graceful degradation — for non-Sanskrit prompts, does it still produce Devanagari?
def metric_ood_recovery(prompt, output):
    is_ood = not any('\u0900' <= c <= '\u097F' for c in prompt)
    if not is_ood:
        return None  # N/A for Sanskrit prompts
    purity = metric_devanagari_purity(output)
    # Good if model still outputs Sanskrit despite OOD prompt
    return round(purity, 3)


# 9. Average token entropy (diversity of token distribution)
def metric_token_entropy(model, tokenizer, device, text):
    if not text.strip():
        return 0.0
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, seq, vocab)
    probs = torch.softmax(logits[0], dim=-1)  # (seq, vocab)
    entropy_per_token = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
    return round(entropy_per_token.mean().item(), 3)


# 10. Relevance — keyword match (extended keyword map)
KEYWORD_MAP = {
    "प्रकृति":   ["पर्वत","नदी","वन","आकाश","वायु","जल","पुष्प","वृक्ष"],
    "महाभारत":   ["धर्म","कुरु","अर्जुन","कृष्ण","भीम","युद्ध","पाण्डव"],
    "धर्म":      ["सत्य","अहिंसा","न्याय","कर्म","मोक्ष","धर्म"],
    "पञ्चभूत":   ["पृथ्वी","आपः","तेजः","वायुः","आकाशः"],
    "ब्रह्म":    ["सत्यम्","आत्मा","चित्","आनन्द","मोक्ष","ब्रह्मन्"],
    "योग":       ["यम","नियम","आसन","प्राणायाम","ध्यान","समाधि"],
    "वर्ण":      ["ब्राह्मण","क्षत्रिय","वैश्य","शूद्र"],
}

def metric_relevance(prompt, output):
    for key, keywords in KEYWORD_MAP.items():
        if key in prompt or key in output[:30]:
            hits = sum(1 for k in keywords if k in output)
            return round(hits / len(keywords), 3)
    return 0.3  # neutral for unmatched prompts

# ──────────────────────────────────────────────
#  COMPOSITE SCORER
# ──────────────────────────────────────────────

def compute_all_metrics(model, tokenizer, device, prompt, output):
    ppl    = metric_perplexity(model, tokenizer, device, output)
    rep    = metric_repetition(output)
    ttr    = metric_ttr(output)
    deva   = metric_devanagari_purity(output)
    sent   = metric_sentence_completion(output)
    length = metric_length(output)
    absorb = metric_prompt_absorption(prompt, output)
    ood    = metric_ood_recovery(prompt, output)
    ent    = metric_token_entropy(model, tokenizer, device, output)
    rel    = metric_relevance(prompt, output)

    # Normalise perplexity: ppl 10→1.0, ppl 500→0.0
    ppl_s = max(0.0, min(1.0, (500 - ppl) / 490))

    # Weighted composite (weights sum to 1.0)
    weights = {
        "relevance":     0.20,
        "devanagari":    0.18,
        "ppl_score":     0.15,
        "repetition":    0.12,
        "absorption":    0.10,
        "sentence_end":  0.10,
        "length":        0.08,
        "ttr":           0.07,
    }

    composite = (
        weights["relevance"]    * rel    +
        weights["devanagari"]   * deva   +
        weights["ppl_score"]    * ppl_s  +
        weights["repetition"]   * rep    +
        weights["absorption"]   * absorb +
        weights["sentence_end"] * sent   +
        weights["length"]       * length +
        weights["ttr"]          * ttr
    ) * 10

    return {
        "perplexity":        ppl,
        "ppl_score":         round(ppl_s, 3),
        "repetition":        rep,
        "ttr":               ttr,
        "devanagari_purity": deva,
        "sentence_complete": sent,
        "length_score":      length,
        "prompt_absorption": absorb,
        "ood_recovery":      ood,
        "token_entropy":     ent,
        "relevance":         rel,
        "composite":         round(composite, 2),
    }

# ──────────────────────────────────────────────
#  PRETTY PRINTER
# ──────────────────────────────────────────────

METRIC_LABELS = {
    "perplexity":        ("Perplexity       ", "↓ lower=better"),
    "repetition":        ("Repetition ratio ", "↑ higher=better"),
    "ttr":               ("Lexical diversity", "↑ higher=better"),
    "devanagari_purity": ("Devanagari purity", "↑ higher=better"),
    "sentence_complete": ("Sentence ending  ", "1=has । or ॥"),
    "length_score":      ("Length adequacy  ", "↑ target≥40tok"),
    "prompt_absorption": ("Prompt absorption", "↑ higher=better"),
    "ood_recovery":      ("OOD recovery     ", "N/A for Sanskrit"),
    "token_entropy":     ("Token entropy    ", "↑ higher=diverse"),
    "relevance":         ("Relevance        ", "↑ higher=better"),
    "composite":         ("★ COMPOSITE SCORE", "/10"),
}

BAR = "─" * 65

def print_result(i, category, prompt, output, metrics):
    print(f"\n{BAR}")
    print(f"  [{category.upper()}] Test {i}")
    print(f"  Prompt : {prompt}")
    print(f"  Output : {output[:100]}{'...' if len(output)>100 else ''}")
    print(f"{BAR}")
    for key, (label, hint) in METRIC_LABELS.items():
        val = metrics.get(key)
        if val is None:
            print(f"  {label} : {'N/A':>8}   ({hint})")
        else:
            print(f"  {label} : {val:>8}   ({hint})")
    print(f"{BAR}")

# ──────────────────────────────────────────────
#  CATEGORY SUMMARY
# ──────────────────────────────────────────────

def print_category_summary(category, results):
    composites = [r["composite"] for r in results]
    avg = round(sum(composites) / len(composites), 2)
    print(f"\n  📦 [{category.upper()}] Average: {avg}/10  "
          f"(min={min(composites)} max={max(composites)})")

# ──────────────────────────────────────────────
#  SAVE REPORT
# ──────────────────────────────────────────────

def save_report(all_results):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"eval_report_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Full report saved → {path}")

# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    model, tokenizer, device = load_model()

    print("=" * 65)
    print("  📊  SANSKRIT LLM — COMPREHENSIVE EVALUATION SUITE")
    print("=" * 65)

    all_results  = {}
    grand_scores = []
    test_counter = 0

    for category, prompts in TEST_SUITE.items():
        cat_results = []

        for prompt in prompts:
            test_counter += 1
            output  = generate(model, tokenizer, device, prompt)
            metrics = compute_all_metrics(model, tokenizer, device, prompt, output)

            print_result(test_counter, category, prompt, output, metrics)

            cat_results.append(metrics)
            grand_scores.append(metrics["composite"])

        print_category_summary(category, cat_results)
        all_results[category] = cat_results

    # ── Grand Summary ──────────────────────────────────────────
    grand_avg = round(sum(grand_scores) / len(grand_scores), 2)
    grand_min = min(grand_scores)
    grand_max = max(grand_scores)

    print("\n" + "=" * 65)
    print("  🏆  GRAND SUMMARY")
    print("=" * 65)
    print(f"  Total prompts evaluated : {test_counter}")
    print(f"  Overall average score   : {grand_avg} / 10")
    print(f"  Best single score       : {grand_max} / 10")
    print(f"  Worst single score      : {grand_min} / 10")

    # ── Category breakdown ────────────────────────────────────
    print("\n  Category breakdown:")
    for category, results in all_results.items():
        avg = round(sum(r["composite"] for r in results) / len(results), 2)
        bar = "█" * int(avg) + "░" * (10 - int(avg))
        print(f"  {category:<22} [{bar}] {avg}/10")

    # ── Metric averages ───────────────────────────────────────
    print("\n  Metric averages across all prompts:")
    metric_keys = ["perplexity","repetition","ttr","devanagari_purity",
                   "sentence_complete","prompt_absorption","relevance"]
    flat = [r for cat in all_results.values() for r in cat]
    for k in metric_keys:
        vals = [r[k] for r in flat if r.get(k) is not None]
        if vals:
            print(f"  {k:<22} : {round(sum(vals)/len(vals), 3)}")

    print("=" * 65)

    save_report(all_results)


if __name__ == "__main__":
    main()