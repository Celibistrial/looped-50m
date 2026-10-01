"""Build the pretraining mix (~2B tokens), a 16k BPE tokenizer, and benchmark eval files.

Outputs (in /kaggle/working): tok.json, train.bin, val_web.bin, val_stories.bin (uint16), evals/*.jsonl, meta.json
"""
import json, os, random, re, time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

OUT = "/kaggle/working"
os.makedirs(f"{OUT}/evals", exist_ok=True)
T0 = time.time()
BPT = 4.2  # rough bytes/token, only used to size text budgets


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


# ---------------- benchmarks: eval sets (held out) + train splits (in mix) ----------------
def mc(ctx, choices, gold):
    return {"ctx": ctx, "choices": choices, "gold": gold}


def arc(r):
    return mc(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"]))


def obqa(r):
    return mc(f"Question: {r['question_stem']}\nAnswer:", [" " + t for t in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"]))


def csqa(r):
    return mc(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]], r["choices"]["label"].index(r["answerKey"]))


def sciq(r):
    ch = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
    return mc(f"{r['support']}\nQuestion: {r['question']}\nAnswer:".lstrip(), [" " + c for c in ch], 3)


def hswag(r):
    pre = lambda s: re.sub(r"\[.*?\]", "", s.replace(" [title]", ". ")).replace("  ", " ").strip()
    ctx = pre(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
    return mc(ctx, [" " + pre(e) for e in r["endings"]], int(r["label"]))


def piqa(r):
    return mc(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"]))


def wino(r):  # partial scoring: context = sentence up to blank with each option, target = rest
    i = r["sentence"].index("_")
    return {"ctxs": [r["sentence"][:i] + r["option1"], r["sentence"][:i] + r["option2"]],
            "cont": " " + r["sentence"][i + 1:].strip(), "gold": int(r["answer"]) - 1}


BENCH = {  # name: (hf id, config, eval split, train split, formatter)
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "test", "train", arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", "train", arc),
    "hellaswag": ("Rowan/hellaswag", None, "validation", "train", hswag),
    "piqa": ("baber/piqa", None, "validation", "train", piqa),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", "train", wino),
    "openbookqa": ("allenai/openbookqa", "main", "test", "train", obqa),
    "sciq": ("allenai/sciq", None, "test", "train", sciq),
    "commonsense_qa": ("tau/commonsense_qa", None, "validation", "train", csqa),
}


def as_text(ex):  # train-split example rendered as plain text with the gold answer
    if "cont" in ex:
        return ex["ctxs"][ex["gold"]] + ex["cont"]
    return ex["ctx"] + ex["choices"][ex["gold"]]


def ngrams(s, n=13):
    w = re.findall(r"\w+", s.lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


eval_grams, bench_train = set(), []
for name, (hf, cfg, ev, tr, fmt) in BENCH.items():
    try:
        dev = [fmt(r) for r in load_dataset(hf, cfg, split=ev)]
        with open(f"{OUT}/evals/{name}.jsonl", "w") as f:
            for ex in dev:
                f.write(json.dumps(ex) + "\n")
                eval_grams |= ngrams(json.dumps(ex))
        trn = [as_text(fmt(r)) for r in load_dataset(hf, cfg, split=tr)]
        bench_train += trn
        log(f"bench {name}: eval {len(dev)}  train {len(trn)}  e.g. {as_text(dev[0])[:120]!r}")
    except Exception as e:
        log(f"bench {name} FAILED: {type(e).__name__}: {e}")

before = len(bench_train)
bench_train = [t for t in bench_train if not (ngrams(t) & eval_grams)]
log(f"decontam: dropped {before - len(bench_train)} / {before} benchmark-train docs with 13-gram eval overlap")
# ponytail: decontamination covers the benchmark train splits only, not the 2B web tokens; scan web too if a judge asks


# ---------------- arithmetic with worked steps (programmatic, no LLM) ----------------
def add_steps(a, b):
    da, db = str(a)[::-1], str(b)[::-1]
    carry, out, steps = 0, [], []
    for i in range(max(len(da), len(db))):
        x, y = int(da[i]) if i < len(da) else 0, int(db[i]) if i < len(db) else 0
        s = x + y + carry
        steps.append(f"{x}+{y}" + (f"+{carry}" if carry else "") + f"={s}, write {s % 10}" + (" carry 1" if s >= 10 else ""))
        carry, out = s // 10, out + [s % 10]
    if carry:
        steps.append("write 1")
    return "; ".join(steps)


def arith(rng):
    k = rng.random()
    if k < 0.45:
        a, b = rng.randint(0, 10 ** rng.randint(1, 5)), rng.randint(0, 10 ** rng.randint(1, 5))
        return f"Q: {a} + {b} = ?\nA: {add_steps(a, b)}. So {a} + {b} = {a + b}."
    if k < 0.7:
        a, b = sorted([rng.randint(0, 10 ** rng.randint(1, 4)) for _ in range(2)], reverse=True)
        return f"Q: {a} - {b} = ?\nA: {a} - {b}: check {b} + {a - b} = {a}. So {a} - {b} = {a - b}."
    if k < 0.9:
        a, b = rng.randint(2, 999), rng.randint(2, 9)
        parts = [(int(d) * 10 ** i) for i, d in enumerate(str(a)[::-1]) if d != "0"][::-1]
        prods = " + ".join(f"{p}*{b}={p * b}" for p in parts)
        return f"Q: {a} * {b} = ?\nA: split {a} = {' + '.join(map(str, parts))}. {prods}. Sum = {a * b}. So {a} * {b} = {a * b}."
    a, b = rng.randint(0, 9999), rng.randint(0, 9999)
    rel = ">" if a > b else "<" if a < b else "="
    return f"Q: Which is larger, {a} or {b}?\nA: {a} {rel} {b}. So the answer is {max(a, b)}."


rng = random.Random(0)
arith_train = [arith(rng) for _ in range(1_600_000)]
rng = random.Random(12345)
with open(f"{OUT}/evals/arith.jsonl", "w") as f:
    seen = set(arith_train)
    n = 0
    while n < 1000:
        s = arith(rng)
        if s in seen:
            continue
        q, a = s.split("\nA:")
        f.write(json.dumps({"prompt": q + "\nA:", "answer": a.rsplit("= ", 1)[-1].rsplit("is ", 1)[-1].rstrip(".")}) + "\n")
        n += 1
log(f"arith: {len(arith_train)} train problems, 1000 held-out")


# ---------------- web / textbook / math / story sources (streamed) ----------------
SOURCES = {  # name: (hf id, config, token budget)
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", 1_000_000_000),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 450_000_000),
    "wikihow": ("HuggingFaceTB/cosmopedia", "wikihow", 50_000_000),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus", 200_000_000),
    "tinystories": ("roneneldan/TinyStories", None, 140_000_000),
}
VAL_DOCS = 5000


def stream(name):
    hf, cfg, budget = SOURCES[name]
    ds = load_dataset(hf, cfg, split="train", streaming=True)
    got = 0
    for r in ds:
        t = r["text"]
        got += len(t.encode())
        yield t
        if got > budget * BPT:
            return


# tokenizer: 16k byte-level BPE, digits split individually (Llama / Qwen practice) so arithmetic is learnable
sample = []
for name in SOURCES:
    it, nbytes = stream(name), 0
    for t in it:
        sample.append(t)
        nbytes += len(t)
        if nbytes > 60e6:
            break
sample += arith_train[:50000] + bench_train[:50000]
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), pre_tokenizers.ByteLevel(add_prefix_space=False)])
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(sample, trainers.BpeTrainer(vocab_size=16384, special_tokens=["<|eot|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok.save(f"{OUT}/tok.json")
EOT = tok.token_to_id("<|eot|>")
del sample
log("tokenizer trained, vocab", tok.get_vocab_size())


def encode_to(f, texts):
    n = 0
    for i in range(0, len(texts), 4096):
        ids = []
        for e in tok.encode_batch(texts[i:i + 4096]):
            ids += e.ids + [EOT]
        np.array(ids, dtype=np.uint16).tofile(f)
        n += len(ids)
    return n


meta = {"sources": {}}
with open(f"{OUT}/train.bin", "wb") as f:
    for name in SOURCES:
        buf, n, val, nb = [], 0, [], 0
        for t in stream(name):
            if len(val) < VAL_DOCS and name in ("fineweb_edu",):
                val.append(t)
                continue
            buf.append(t)
            if len(buf) == 50000:
                n += encode_to(f, buf)
                buf = []
        n += encode_to(f, buf)
        if val:
            with open(f"{OUT}/val_web.bin", "wb") as vf:
                vt = encode_to(vf, val)
            meta["val_web_bytes_per_token"] = sum(len(t.encode()) for t in val) / vt
        meta["sources"][name] = n
        log(f"{name}: {n / 1e6:.1f}M tokens")
    n = encode_to(f, bench_train * 3)
    meta["sources"]["benchmark_train_x3"] = n
    n = encode_to(f, arith_train)
    meta["sources"]["arithmetic"] = n
    log(f"bench x3 {meta['sources']['benchmark_train_x3'] / 1e6:.1f}M, arith {n / 1e6:.1f}M tokens")

sv = [r["text"] for r in load_dataset("roneneldan/TinyStories", split="validation")]
with open(f"{OUT}/val_stories.bin", "wb") as vf:
    vt = encode_to(vf, sv)
meta["val_stories_bytes_per_token"] = sum(len(t.encode()) for t in sv) / vt
meta["train_tokens"] = int(os.path.getsize(f"{OUT}/train.bin") // 2)
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
log("DONE", json.dumps(meta))
