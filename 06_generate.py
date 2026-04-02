"""
06_generate.py
==============
STEP 6 — Generate Sanskrit text using the trained model.

Usage:
    python 06_generate.py                         # Interactive mode
    python 06_generate.py --prompt "रामः"         # Single prompt
    python 06_generate.py --prompts-file p.txt    # Batch mode
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logging_utils import log, log_section, log_stats
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEFAULT_GEN = {
    "max_new_tokens": 200, "temperature": 0.8, "top_k": 50, "top_p": 0.92,
    "repetition_penalty": 1.3, "do_sample": True, "num_beams": 1,
    "num_return_sequences": 1, "early_stopping": True,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--prompts-file", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--num-beams", type=int, default=1)
    p.add_argument("--num-return", type=int, default=1)
    p.add_argument("--no-sample", action="store_true")
    return p.parse_args()


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


def load_model_and_tokenizer(checkpoint):
    log(f"Loading model from: {checkpoint}", "📂")
    if not os.path.exists(checkpoint):
        log(f"Not found: {checkpoint}", "❌"); sys.exit(1)
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint)
    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    log(f"Model loaded on {device}", "✅")
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, gen_config):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, **gen_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    results = []
    for seq in output_ids:
        full = tokenizer.decode(seq, skip_special_tokens=True).strip()
        new  = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        results.append({"full": full, "generated": new, "n_tokens": len(seq) - input_len})
    return results


def print_generation(prompt, results):
    print("\n" + "─" * 60)
    print(f"📝 Prompt    : {prompt}")
    print("─" * 60)
    for i, r in enumerate(results):
        if len(results) > 1:
            print(f"\n  [Sequence {i+1}]")
        print(f"  Generated ({r['n_tokens']} tokens): {r['generated']}")
        print(f"  Full text : {r['full']}")
    print("─" * 60)


def interactive_mode(model, tokenizer, device, gen_config):
    log_section("Interactive Sanskrit Generation")
    print("  Type a Sanskrit prompt in Devanagari. 'quit' to exit.\n")
    while True:
        try:
            prompt = input("🕉️  Prompt: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break
            print_generation(prompt, generate_text(model, tokenizer, device, prompt, gen_config))
        except KeyboardInterrupt:
            break
    log("नमस्ते!", "👋")


def batch_mode(model, tokenizer, device, prompts_file, gen_config):
    with open(prompts_file, encoding="utf-8") as f:
        prompts = [l.strip() for l in f if l.strip()]
    log(f"Loaded {len(prompts)} prompts", "📄")
    output_lines = []
    for i, prompt in enumerate(prompts, 1):
        log(f"Generating {i}/{len(prompts)}", "⚡")
        results = generate_text(model, tokenizer, device, prompt, gen_config)
        print_generation(prompt, results)
        for r in results:
            output_lines += [f"PROMPT: {prompt}", f"OUTPUT: {r['full']}", ""]
    out = prompts_file.replace(".txt", "_generated.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log(f"Saved: {out}", "💾")


def main():
    args = parse_args()
    log_section("STEP 6 — Sanskrit Text Generation")

    gen_config = DEFAULT_GEN.copy()
    gen_config.update({
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_k": args.top_k, "top_p": args.top_p, "num_beams": args.num_beams,
        "num_return_sequences": args.num_return, "do_sample": not args.no_sample,
    })
    if args.num_beams > 1:
        gen_config.update({"do_sample": False, "temperature": 1.0})

    checkpoint = args.checkpoint or find_best_checkpoint()
    model, tokenizer, device = load_model_and_tokenizer(checkpoint)

    if args.prompts_file:
        batch_mode(model, tokenizer, device, args.prompts_file, gen_config)
    elif args.prompt:
        print_generation(args.prompt, generate_text(model, tokenizer, device, args.prompt, gen_config))
    else:
        interactive_mode(model, tokenizer, device, gen_config)


if __name__ == "__main__":
    main()