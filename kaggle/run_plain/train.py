"""JAX/TPU trainer + evaluator for the 50M plain and recurrent-depth (looped) LMs.

MODE = "plain":  13 x d512 layers, ~47.6M params
MODE = "looped": prelude 2 + core 4 (looped r times) + coda 2, d640, ~47.9M params.
                 Trained with r ~ U{1..6}; evaluated at r in EVAL_R (test-time compute scaling).
"""
import glob, json, math, os, re, subprocess, sys, time
from functools import partial

MODE = os.environ.get("MODE", "plain")
SMOKE = bool(os.environ.get("SMOKE"))  # tiny local CPU run to catch bugs
TOKENS = 2_000_000_000
OUT = os.environ.get("OUT", "/kaggle/working")

for mod in ("tokenizers", "optax"):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=True)
import jax, jax.numpy as jnp, numpy as np, optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tokenizers import Tokenizer

T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f}m]", *a, flush=True)


DATA = os.path.dirname(glob.glob(os.environ.get("INPUT", "/kaggle/input") + "/**/train.bin", recursive=True)[0])
meta = json.load(open(f"{DATA}/meta.json"))
tok = Tokenizer.from_file(f"{DATA}/tok.json")
EOT = tok.token_to_id("<|eot|>")
V, CTX, B = tok.get_vocab_size(), 2048, 128  # 262k tokens per step
if SMOKE:
    CTX, B, TOKENS = 256, 8, 8 * 257 * 12
CFG = {
    "plain": dict(d=512, heads=8, kv=2, ffn=1536, pre=13, core=0, coda=0),
    "looped": dict(d=640, heads=10, kv=2, ffn=1920, pre=2, core=4, coda=2),
}[MODE]
D, H, KV, F = CFG["d"], CFG["heads"], CFG["kv"], CFG["ffn"]
HD = D // H
TRAIN_R = [1, 2, 3, 4, 5, 6] if CFG["core"] else [1]
EVAL_R = [1, 2, 4, 6, 8, 12] if CFG["core"] else [1]
bf16, f32 = jnp.bfloat16, jnp.float32

mesh = Mesh(np.array(jax.devices()), ("data",))
DSH, REP = NamedSharding(mesh, P("data")), NamedSharding(mesh, P())
log(MODE, CFG, jax.devices())


# ---------------- model ----------------
def init_stack(key, n):
    k = jax.random.split(key, 3)
    nrm = lambda k, s: 0.02 * jax.random.normal(k, (n, *s), f32)
    return {
        "n1": jnp.ones((n, D)), "n2": jnp.ones((n, D)), "qn": jnp.ones((n, HD)), "kn": jnp.ones((n, HD)),
        "wq": nrm(k[0], (D, H * HD)), "wkv": nrm(k[1], (D, 2 * KV * HD)), "wo": jnp.zeros((n, H * HD, D)),
        "wgu": nrm(k[2], (D, 2 * F)), "wd": jnp.zeros((n, F, D)),
    }


def init_params(key):
    k = jax.random.split(key, 4)
    p = {"emb": 0.02 * jax.random.normal(k[0], (V, D), f32), "nf": jnp.ones(D), "pre": init_stack(k[1], CFG["pre"])}
    if CFG["core"]:
        p |= {"core": init_stack(k[2], CFG["core"]), "coda": init_stack(k[3], CFG["coda"]), "inj": jnp.ones(D)}
    return p


def rms(x, w):
    x32 = x.astype(f32)
    return (x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, -1, keepdims=True) + 1e-6)).astype(x.dtype) * w


inv = 1.0 / (10000 ** (np.arange(0, HD, 2) / HD))
ang = np.outer(np.arange(CTX), inv)
COS, SIN = jnp.asarray(np.cos(ang), bf16), jnp.asarray(np.sin(ang), bf16)


def rope(x):  # B,T,H,hd
    T = x.shape[1]
    c, s = COS[None, :T, None], SIN[None, :T, None]
    x1, x2 = jnp.split(x, 2, -1)
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)


def block(x, w):
    Bn, T, _ = x.shape
    h = rms(x, w["n1"])
    q = (h @ w["wq"]).reshape(Bn, T, H, HD)
    kv = (h @ w["wkv"]).reshape(Bn, T, 2, KV, HD)
    q, k, v = rope(rms(q, w["qn"])), rope(rms(kv[:, :, 0], w["kn"])), kv[:, :, 1]
    k, v = jnp.repeat(k, H // KV, 2), jnp.repeat(v, H // KV, 2)
    s = jnp.einsum("bthd,bshd->bhts", q, k).astype(f32) / math.sqrt(HD)
    s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -1e30)
    y = jnp.einsum("bhts,bshd->bthd", jax.nn.softmax(s, -1).astype(bf16), v).reshape(Bn, T, H * HD)
    x = x + y @ w["wo"]
    g, u = jnp.split(rms(x, w["n2"]) @ w["wgu"], 2, -1)
    return x + (jax.nn.silu(g) * u) @ w["wd"]


def run(stack, x):
    return jax.lax.scan(jax.checkpoint(lambda x, w: (block(x, w), None)), x, stack)[0]


def forward(params, idx, r):
    p = jax.tree.map(lambda a: a.astype(bf16), params)
    x = run(p["pre"], p["emb"][idx])
    if CFG["core"]:
        h = jnp.zeros_like(x)
        for _ in range(r):  # recurrent depth with input injection
            h = run(p["core"], rms(h, p["inj"]) + x)
        x = run(p["coda"], h)
    logits = (rms(x, p["nf"]) @ p["emb"].T).astype(f32)
    return 30.0 * jnp.tanh(logits / 30.0)  # Gemma-2 soft-cap


def loss_fn(params, batch, r):
    logits = forward(params, batch[:, :-1], r)
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:]).mean()


# ---------------- optimizer: Muon (block matrices) + AdamW (embedding, norms) ----------------
def ortho(G):  # Newton-Schulz quintic (Keller Jordan), per matrix
    a, b, c = 3.4445, -4.7750, 2.0315
    X = (G / (jnp.linalg.norm(G) + 1e-7)).astype(bf16)
    tr = X.shape[0] > X.shape[1]
    X = X.T if tr else X
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    X = X.T if tr else X
    return X.astype(f32) * math.sqrt(max(1.0, G.shape[1] / G.shape[0]))


def muon(beta=0.95):
    def init(params):
        return jax.tree.map(jnp.zeros_like, params)

    def update(g, m, params=None):
        m = jax.tree.map(lambda m, g: beta * m + g, m, g)
        u = jax.tree.map(lambda m, g: jax.vmap(ortho)(g + beta * m), m, g)  # nesterov, vmapped over layer axis
        return u, m

    return optax.GradientTransformation(init, update)


STEPS = min(TOKENS, os.path.getsize(f"{DATA}/train.bin") // 2) // (B * (CTX + 1))  # at most 1 epoch


def wsd(peak):  # warmup 2%, stable, linear cooldown over last 25%
    w, c = max(1, int(0.02 * STEPS)), int(0.75 * STEPS)
    return lambda s: peak * jnp.minimum(jnp.minimum(s / w, 1.0), jnp.clip((STEPS - s) / (STEPS - c), 0.0, 1.0))


MUON_KEYS = {"wq", "wkv", "wo", "wgu", "wd"}
labels = lambda p: jax.tree_util.tree_map_with_path(lambda path, _: "muon" if path[-1].key in MUON_KEYS else "adam", p)
opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.multi_transform({
        "muon": optax.chain(muon(), optax.scale_by_learning_rate(wsd(0.02))),
        "adam": optax.adamw(wsd(3e-3), b1=0.9, b2=0.95, weight_decay=0.0),
    }, labels),
)


@partial(jax.jit, static_argnums=3, donate_argnums=(0, 1))
def train_step(params, opt_state, batch, r):
    loss, g = jax.value_and_grad(loss_fn)(params, batch, r)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, loss


@partial(jax.jit, static_argnums=2)
def token_logp(params, x, r):  # x: B,L+1 -> logp of x[:,1:]
    lp = jax.nn.log_softmax(forward(params, x[:, :-1], r), -1)
    return jnp.take_along_axis(lp, x[:, 1:, None], -1)[..., 0]


@partial(jax.jit, static_argnums=3)
def next_logits(params, buf, lens, r):
    lg = forward(params, buf, r)
    return lg[jnp.arange(buf.shape[0]), lens - 1]


# ---------------- evals ----------------
def val_loss(params, path, r, n_win=1024):
    d = np.memmap(path, dtype=np.uint16, mode="r")
    n_win = max(B, min(n_win, len(d) // (CTX + 1)) // B * B)
    x = np.asarray(d[: n_win * (CTX + 1)], np.int32).reshape(n_win, CTX + 1)
    return float(np.mean([np.asarray(token_logp(params, jax.device_put(x[i:i + B], DSH), r)).mean() for i in range(0, n_win, B)])) * -1


def score_rows(params, rows, r, L=256):  # rows: (ctx_ids, cont_ids) -> summed logp of cont
    out = []
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        x, spans = np.zeros((B, L + 1), np.int32), []
        for j, (c, t) in enumerate(chunk):
            s = (c + t)[-(L + 1):]
            x[j, :len(s)] = s
            spans.append((len(s) - 1 - len(t), len(s) - 1))
        lp = np.asarray(token_logp(params, jax.device_put(x, DSH), r))
        out += [lp[j, a:b].sum() for j, (a, b) in enumerate(spans)]
    return out


enc = lambda s: tok.encode(s).ids


def bench(params, r):
    res = {}
    for f in sorted(glob.glob(f"{DATA}/evals/*.jsonl")):
        name = os.path.basename(f)[:-6]
        if name == "arith":
            continue
        exs = [json.loads(l) for l in open(f)]
        rows, idx = [], []
        for ex in exs:
            if "cont" in ex:  # winogrande partial scoring
                rows += [(enc(c), enc(ex["cont"])) for c in ex["ctxs"]]
                idx.append((len(ex["ctxs"]), [1] * len(ex["ctxs"])))
            else:
                rows += [(enc(ex["ctx"]), enc(ch)) for ch in ex["choices"]]
                idx.append((len(ex["choices"]), [len(ch.encode()) for ch in ex["choices"]]))
        sc, k, acc, accn = score_rows(params, rows, r), 0, 0, 0
        for ex, (n, lens) in zip(exs, idx):
            s = np.array(sc[k:k + n])
            acc += int(s.argmax() == ex["gold"])
            accn += int((s / np.array(lens)).argmax() == ex["gold"])
            k += n
        res[name] = {"acc": acc / len(exs), "acc_norm": accn / len(exs), "n": len(exs)}
    return res


def generate(params, prompts, r, max_new=160, temp=0.0, seed=0, L=256):
    ids = [enc(p)[-(L - max_new):] for p in prompts]
    outs = []
    key = jax.random.PRNGKey(seed)
    for i in range(0, len(ids), B):
        chunk = ids[i:i + B]
        buf, lens = np.zeros((B, L), np.int32), np.ones(B, np.int32)
        for j, s in enumerate(chunk):
            buf[j, :len(s)], lens[j] = s, len(s)
        start, done = lens.copy(), np.zeros(B, bool)
        for _ in range(max_new):
            lg = next_logits(params, jax.device_put(buf, DSH), jax.device_put(lens, DSH), r)
            if temp == 0:
                nxt = np.asarray(jnp.argmax(lg, -1))
            else:
                key, sk = jax.random.split(key)
                kth = jnp.sort(lg, -1)[:, -50][:, None]
                nxt = np.asarray(jax.random.categorical(sk, jnp.where(lg < kth, -1e30, lg) / temp))
            live = ~done & (lens < L)
            buf[np.arange(B)[live], lens[live]] = nxt[live]
            lens[live] += 1
            done |= nxt == EOT
            if done[:len(chunk)].all():
                break
        outs += [tok.decode([t for t in buf[j, start[j]:lens[j]].tolist() if t != EOT]) for j in range(len(chunk))]
    return outs


def arith_acc(params, r):
    exs = [json.loads(l) for l in open(f"{DATA}/evals/arith.jsonl")]
    gens = generate(params, [e["prompt"] for e in exs], r)
    ok = 0
    for e, g in zip(exs, gens):
        g = g.split("\nQ:")[0]
        m = re.findall(r"(?:= |is )(-?\d+)", g.split("So ")[-1]) if "So " in g else []
        ok += int(bool(m) and m[-1] == e["answer"])
    return ok / len(exs), gens[:5]


# ---------------- train ----------------
params = jax.device_put(init_params(jax.random.PRNGKey(0)), REP)
nparams = sum(x.size for x in jax.tree.leaves(params))
log(f"{nparams / 1e6:.2f}M params, {STEPS} steps of {B}x{CTX}, {TOKENS / 1e9:.1f}B tokens")
assert nparams <= 50_000_000
opt_state = opt.init(params)

data = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r")
starts = np.random.default_rng(0).permutation(len(data) // (CTX + 1))[: STEPS * B] * (CTX + 1)
assert len(starts) == STEPS * B, f"not enough data: {len(data)} tokens"
rng = np.random.default_rng(1)
logf = open(f"{OUT}/log.csv", "w")
logf.write("step,tokens,hours,train_loss,val_web,val_stories\n")
t_last, tok_last = time.time(), 0
for step in range(STEPS):
    idx = starts[step * B:(step + 1) * B]
    batch = np.stack([data[i:i + CTX + 1] for i in idx]).astype(np.int32)
    r = int(rng.choice(TRAIN_R))
    params, opt_state, loss = train_step(params, opt_state, jax.device_put(batch, DSH), r)
    if step % 50 == 0 or step == STEPS - 1:
        loss = float(loss)
        dt, done_tok = time.time() - t_last, (step + 1) * B * CTX
        tps = (done_tok - tok_last) / dt
        t_last, tok_last = time.time(), done_tok
        eta = (STEPS - step) * B * CTX / tps / 60
        log(f"step {step}/{STEPS} loss {loss:.4f} r={r} {tps / 1e3:.0f}k tok/s eta {eta:.0f}m")
        assert math.isfinite(loss), "loss diverged"
    if step % 500 == 0 or step == STEPS - 1:
        rv = 4 if CFG["core"] else 1
        vw, vs = val_loss(params, f"{DATA}/val_web.bin", rv, B), val_loss(params, f"{DATA}/val_stories.bin", rv, B)
        logf.write(f"{step},{(step + 1) * B * CTX},{(time.time() - T0) / 3600:.4f},{float(loss):.4f},{vw:.4f},{vs:.4f}\n")
        logf.flush()
        log(f"  val_web {vw:.4f} val_stories {vs:.4f} (r={rv})")

flat = {jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_flatten_with_path(params)[0]}
np.savez(f"{OUT}/params_{MODE}.npz", **flat)
train_hours = (time.time() - T0) / 3600
log("saved params; evaluating")

# ---------------- final eval ----------------
results = {"mode": MODE, "params": int(nparams), "cfg": CFG, "tokens": STEPS * B * CTX, "train_hours": train_hours,
           "hardware": f"Kaggle TPU v5e-8 ({len(jax.devices())} chips)", "by_r": {}}
for r in EVAL_R:
    vw, vs = val_loss(params, f"{DATA}/val_web.bin", r), val_loss(params, f"{DATA}/val_stories.bin", r)
    ra = {"val_web_loss": vw, "val_web_ppl": math.exp(vw), "val_web_bpb": vw / math.log(2) / meta["val_web_bytes_per_token"],
          "val_stories_loss": vs, "val_stories_ppl": math.exp(vs), "val_stories_bpb": vs / math.log(2) / meta["val_stories_bytes_per_token"]}
    ra["bench"] = bench(params, r)
    ra["arith_acc"], ra["arith_samples"] = arith_acc(params, r)
    results["by_r"][r] = ra
    log(f"r={r}: web ppl {ra['val_web_ppl']:.2f} bpb {ra['val_web_bpb']:.4f} | stories ppl {ra['val_stories_ppl']:.2f} | arith {ra['arith_acc']:.3f} | "
        + " ".join(f"{k}={v['acc_norm']:.3f}" for k, v in ra["bench"].items()))
    json.dump(results, open(f"{OUT}/results_{MODE}.json", "w"), indent=1)

prompts = ["Once upon a time, there was a little girl named Lily who", "Photosynthesis is the process by which",
           "Q: 4821 + 3967 = ?\nA:", "The three states of matter are", "How to make a paper airplane:\n1."]
rs = EVAL_R[-3] if CFG["core"] else 1
results["samples"] = {p: generate(params, [p], rs, max_new=120, temp=0.8)[0] for p in prompts}
json.dump(results, open(f"{OUT}/results_{MODE}.json", "w"), indent=1)
for p, s in results["samples"].items():
    log(f"--- {p!r}\n{s}")
log("ALL DONE")
