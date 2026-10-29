#!/usr/bin/env python3
"""Sample the injection / held-out cohorts, build their texts, tokenize, place copies.

Deterministic given --seed: every candidate list is sorted before a draw and
each stage derives its own RNG from the master seed. The randomization unit is
the matched pair: within cohort x stratum, documents are sorted by (length
quartile, pre-treatment bpb), paired adjacently, and ONE coin flip per pair
decides which member is injected. Manipulations (dose r, target length) are
assigned per pair, so the held-out partner carries the same label.

Subcommands
  gen-samples  GPU: unconditional samples from the pre-treatment checkpoint
               (anchor stratum "model_samples"); run before `build`
  build        everything else: sampling, L truncation, shuffled/random anchors,
               tokenization, copy positions, summary
  refill       after a later scan excludes documents: drop every pair that
               contains one, and refill each slot with a new pair from the
               same cohort x stratum drawn from the spare candidates that
               passed the full audit. The replacement inherits the slot
               (pair_id, r, target_len, split) and the dropped inject
               document's copy positions, so nothing else in the set moves.
               Spares: eligible candidates not yet in the manifest or null
               pool; the domain's null pool is the reserve of last resort
               (SE and wiki B have no other spares).

Inputs (defaults under the collection/ directory of the data root, see
olmo_twin/paths.py; audit_v2/ and wiki_edited/ are written by the scripts in
collection/):
  stageG/scoring_manifest.csv.gz      clean non-members with raw_text_path
  logprobs_stage1/scores.csv.gz       pre-treatment n_tokens, bpb
  wiki_edited/edited_group.csv.gz     near-duplicate members with strata
  wiki_edited/logprobs_stage1/...     their pre-treatment scores
  postcutoff/pool_parsed_merged.csv.gz  arXiv LaTeX / PDF raw text paths

Outputs (--out-dir, default: the bundle/ directory of the data root):
  manifest.csv.gz   one row per document in a cohort (both arms)
  copies.csv.gz     one row per injected COPY with decile and position_frac
  null_pool.csv.gz  never-injected documents per domain
  texts/            texts that differ from the source file (L prefixes, anchors, F LaTeX)
  tokens/<cohort>/<doc>.npy   token_ids (int32) incl. trailing EOS
  summary.json      per-cohort counts, injected tokens, share of the stage-2 run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# repository root on sys.path, so that `olmo_twin` imports from a checkout
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

W = paths.COLLECTION   # collection working directory (audit_v2/, wiki_edited/, postcutoff/)
EOS_NOTE = "add_special_tokens=False + trailing EOS (as the training stream separates documents)"

# ---------------------------------------------------------------------------
# cohort specification: per-arm sizes
# ---------------------------------------------------------------------------
ARXIV_MIN, ARXIV_MAX, SHORT_MIN = 1024, 32768, 256
L_LEVELS = [512, 1024, 2048, 4096, 8192]
D_LEVELS = [2, 4, 8, 16]
X_STRATA = ["model_samples", "sentence_shuffled", "word_shuffled", "random_tokens"]
X_TARGET_TOKENS = 1024
E_STRATA = ["lt10", "10-25", "25-50", "50-75", "ge75"]

SPEC = {
    # cohort: dict(source, per_arm, ...)
    "L":     dict(source="arxiv", per_arm=250, question="amount of injected text (prefix length)"),
    "M":     dict(source="arxiv", per_arm=2000, question="headline single-exposure effect; covariates; temporal novelty by month"),
    "D":     dict(source="arxiv", per_arm=400, question="dose response r in {2,4,8,16}"),
    "D32":   dict(source="arxiv", per_arm=40, r=32, question="dose curve tail"),
    "DB":    dict(source="arxiv", per_arm=60, r=4, burst=True, question="burst vs spread exposure"),
    "F":     dict(source="arxiv", per_arm=300, question="LaTeX-derived text injected instead of PDF-derived; format transfer"),
    "X":     dict(source="arxiv_x", per_arm=50, question="surprisal anchors"),
    "W-new": dict(source="wikiA", per_arm=1000, question="encyclopedic text created after the cutoff"),
    "W-gap": dict(source="wikiB", per_arm=600, question="pre-cutoff pages absent from the wiki snapshot"),
    "S-se":  dict(source="se", per_arm=1000, question="forum register, short documents"),
    "E0":    dict(source="edited_identical", per_arm=300, question="second epoch of text already trained on"),
    "E0-D":  dict(source="edited_identical", per_arm=200, r=4, question="repeated re-exposure of known text"),
    "E":     dict(source="edited", per_arm=300, question="near-duplicate members: novel vs shared tokens by novelty stratum"),
}
NULL = {"arxiv": 3000, "wikiA": 1500, "wikiB": 600, "se": 665, "edited": 2000}
X_SOURCE_DOCS = 100   # arXiv docs consumed per shuffled stratum (50 inject + 50 holdout)
M_MONTH_FLOOR_PER_CAT = 20
DEV_FRAC = 0.20
TRIM = 0.01           # first/last 1% of the run receives no copies
BURST_WINDOW = 0.01


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rd(p: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def wr(p: Path, rows: list[dict], cols: list[str] | None = None) -> None:
    cols = cols or list(rows[0].keys())
    with gzip.open(p, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def safe(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


def sub_rng(seed: int, name: str) -> random.Random:
    return random.Random(f"{seed}:{name}")


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------
def make_pairs(docs: list[dict], rng: random.Random, cohort: str, stratum: str,
               pair_counter: list[int]) -> list[dict]:
    """docs: dicts with n_tokens, bpb_pre. Sort by (length quartile, bpb), pair
    adjacent, flip a coin per pair. Returns rows with arm / pair_id set."""
    if len(docs) < 2:
        return []
    docs = sorted(docs, key=lambda d: d["doc_id"])
    # pair on the AS-INJECTED text when its pre-treatment score is known
    # (L prefixes, F LaTeX form); otherwise on the source document
    lens = np.array([d.get("pair_n_tokens") or d["n_tokens"] for d in docs])
    q = np.quantile(lens, [0.25, 0.5, 0.75])
    for d in docs:
        d["length_bin"] = int(np.searchsorted(q, d.get("pair_n_tokens") or d["n_tokens"], side="right"))
        d["pair_bpb"] = d.get("pair_bpb") if d.get("pair_bpb") is not None else d["bpb_pre"]
    docs.sort(key=lambda d: (d["length_bin"], d["pair_bpb"], d["doc_id"]))
    out = []
    for i in range(0, len(docs) - 1, 2):
        a, b = docs[i], docs[i + 1]
        if rng.random() < 0.5:
            a, b = b, a
        pair_counter[0] += 1
        pid = f"{cohort}-{pair_counter[0]:05d}"
        for d, arm in ((a, "inject"), (b, "holdout")):
            out.append({**d, "cohort": cohort, "stratum": stratum, "arm": arm, "pair_id": pid})
    return out


def take(pool: dict[str, dict], ids: list[str]) -> list[dict]:
    got = [pool.pop(i) for i in ids if i in pool]
    return got


def stratified_draw(pool: dict[str, dict], key, targets: dict, rng: random.Random) -> list[dict]:
    """Draw `targets[stratum]` documents per stratum (sorted ids, rng.sample)."""
    by = defaultdict(list)
    for d in pool.values():
        by[key(d)].append(d["doc_id"])
    chosen = []
    for s, n in sorted(targets.items()):
        ids = sorted(by.get(s, []))
        chosen += rng.sample(ids, min(n, len(ids)))
    return take(pool, chosen)


# ---------------------------------------------------------------------------
# texts
# ---------------------------------------------------------------------------
PARA_RE = re.compile(r"\n+")   # peS2o-style text separates paragraphs by single newlines
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")


def truncate_paragraph(text: str, tok, target: int) -> str:
    """Prefix ending at the paragraph boundary whose token count is nearest target."""
    bounds = [m.end() for m in PARA_RE.finditer(text)] + [len(text)]
    best, best_gap = None, None
    lo, hi = 0, len(bounds) - 1
    # token counts are monotone in prefix length: binary search the boundary
    while lo <= hi:
        mid = (lo + hi) // 2
        n = len(tok(text[:bounds[mid]], add_special_tokens=False)["input_ids"])
        gap = abs(n - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = bounds[mid], gap
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return text[:best].rstrip() + "\n"


def sentence_shuffle(text: str, rng: random.Random) -> str:
    sents = [s for s in SENT_RE.split(text.replace("\n", " ")) if s.strip()]
    rng.shuffle(sents)
    return " ".join(sents) + "\n"


def word_shuffle(text: str, rng: random.Random) -> str:
    words = text.split()
    rng.shuffle(words)
    return " ".join(words) + "\n"


# ---------------------------------------------------------------------------
# gen-samples (GPU)
# ---------------------------------------------------------------------------
def cmd_gen_samples(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = args.out_dir / "texts" / "X_model_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16).cuda().eval()
    n = 2 * SPEC["X"]["per_arm"]
    torch.manual_seed(args.seed)
    rows = []
    bs = 10
    for b in range(0, n, bs):
        inp = torch.full((bs, 1), tok.eos_token_id, device="cuda")
        with torch.no_grad():
            gen = model.generate(inp, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                                 max_new_tokens=X_TARGET_TOKENS, min_new_tokens=X_TARGET_TOKENS,
                                 pad_token_id=tok.eos_token_id)
        for j in range(bs):
            ids = gen[j, 1:].tolist()
            text = tok.decode(ids, skip_special_tokens=True)
            doc_id = f"X:model_samples:{b + j:03d}"
            p = out_dir / f"{safe(doc_id)}.txt"
            p.write_text(text, encoding="utf-8")
            np.save(out_dir / f"{safe(doc_id)}.ids.npy", np.asarray(ids, dtype=np.int32))
            rows.append({"doc_id": doc_id, "text_path": str(p), "ids_path": str(out_dir / f"{safe(doc_id)}.ids.npy"),
                         "n_tokens_generated": len(ids)})
        log(f"  generated {min(b + bs, n)}/{n}")
    wr(args.out_dir / "texts" / "X_model_samples.csv.gz", rows)
    log(f"wrote {n} model samples ({args.revision}, T=1.0, {X_TARGET_TOKENS} tokens each, seed {args.seed})")
    return 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def cmd_build(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos = tok.eos_token_id
    out = args.out_dir
    (out / "texts").mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # ---- load candidates -------------------------------------------------
    clean = rd(args.audit_dir / "stageG" / "scoring_manifest.csv.gz")
    sc = {r["doc_id"]: r for r in rd(args.audit_dir / "logprobs_stage1" / "scores.csv.gz") if r["status"] == "ok"}
    arx = {r["doc_id"]: r for r in rd(args.arxiv_pool)}
    pools: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in clean:
        s = sc.get(r["doc_id"])
        if not s:
            continue
        d = {"doc_id": r["doc_id"], "group": r["group"], "month": r["month"], "category_group": r["category_group"],
             "title": r["title"], "text_path": r["raw_text_path"], "text_form": r["text_form"],
             "n_tokens": int(s["n_tokens"]), "bpb_pre": float(s["bpb"]), "alt_text_path": "", "alt_text_form": ""}
        if r["group"] == "postcutoff":
            if not (ARXIV_MIN <= d["n_tokens"] <= ARXIV_MAX):
                continue
            a = arx.get(r["doc_id"], {})
            if r["text_form"] == "pes2o_style_pdf" and a.get("latex_raw_status") == "ok":
                d["alt_text_path"], d["alt_text_form"] = a["latex_raw_path"], "redpajama_latex"
            elif r["text_form"] == "redpajama_latex" and a.get("pdf_raw_status") == "ok":
                d["alt_text_path"], d["alt_text_form"] = a["pdf_raw_path"], "pes2o_style_pdf"
            pools["arxiv"][d["doc_id"]] = d
        else:
            if d["n_tokens"] < SHORT_MIN:
                continue
            pools[r["group"]][d["doc_id"]] = d
    ed_sc = {r["doc_id"]: r for r in rd(args.edited_dir / "logprobs_stage1" / "scores.csv.gz") if r["status"] == "ok"}
    for r in rd(args.edited_dir / "edited_group.csv.gz"):
        s = ed_sc.get(r["doc_id"])
        if not s or int(s["n_tokens"]) < SHORT_MIN:
            continue
        d = {"doc_id": r["doc_id"], "group": "edited", "month": r["last_revision"][:7], "category_group": "",
             "title": r["title"], "text_path": r["raw_text_path"], "text_form": "dolma_wiki",
             "n_tokens": int(s["n_tokens"]), "bpb_pre": float(s["bpb"]), "alt_text_path": "", "alt_text_form": "",
             "novelty_stratum": "identical" if int(r["n_words_novel"]) == 0 else r["stratum"],
             "frac_novel_tokens": float(r["frac_novel_tokens"]), "mask_path": r["mask_path"],
             "ancestor_record_id": r["ancestor_record_id"]}
        pools["edited"][d["doc_id"]] = d
    log("eligible: " + ", ".join(f"{k} {len(v):,}" for k, v in sorted(pools.items())))

    rows: list[dict] = []
    pair_counter = [0]
    A = pools["arxiv"]
    pair_scores: dict[str, dict] = {}
    if args.pair_scores and args.pair_scores.exists():
        for r in rd(args.pair_scores):
            if r["status"] == "ok":
                pair_scores[r["doc_id"]] = r
        log(f"pair scores for as-injected texts: {len(pair_scores):,}")

    def attach_pair_score(d: dict, *keys: str) -> None:
        for k in keys:
            sc_ = pair_scores.get(k)
            if sc_:
                d["pair_bpb"], d["pair_n_tokens"] = float(sc_["bpb"]), int(sc_["n_tokens"])
                return

    # ---- L: long papers, target length per pair BEFORE pairing --------------
    rng = sub_rng(seed, "L")
    longs = sorted(i for i, d in A.items() if d["n_tokens"] >= 8192)
    n_l = 2 * SPEC["L"]["per_arm"]
    chosen = take(A, rng.sample(longs, min(n_l, len(longs))))
    per_level = len(chosen) // len(L_LEVELS)
    rng.shuffle(chosen)
    for k, lvl in enumerate(L_LEVELS):
        grp = chosen[k * per_level:(k + 1) * per_level]
        for d in grp:
            d["target_len"] = lvl
            attach_pair_score(d, f"L:{d['doc_id']}:inj")
        rows += make_pairs(grp, sub_rng(seed, f"L:{lvl}"), "L", f"len{lvl}", pair_counter)

    # ---- M: month-uniform, category proportional with floor ---------------
    rng = sub_rng(seed, "M")
    n_m = 2 * SPEC["M"]["per_arm"]
    months = sorted({d["month"] for d in A.values()})
    avail = {m: sorted(i for i, d in A.items() if d["month"] == m) for m in months}
    per_month = {m: 0 for m in months}
    remaining, open_months = n_m, set(months)
    while remaining > 0 and open_months:          # water-filling toward uniform
        share = max(1, remaining // len(open_months))
        for m in sorted(open_months):
            room = len(avail[m]) - per_month[m]
            add = min(share, room, remaining)
            per_month[m] += add
            remaining -= add
            if per_month[m] >= len(avail[m]):
                open_months.discard(m)
            if remaining <= 0:
                break
    for m in months:
        ids = avail[m]
        cats = Counter(A[i]["category_group"] for i in ids)
        target_m = per_month[m]
        # proportional with floor, then adjust to total
        alloc = {c: max(M_MONTH_FLOOR_PER_CAT if cats[c] >= M_MONTH_FLOOR_PER_CAT else cats[c],
                        round(target_m * cats[c] / len(ids))) for c in cats}
        for c in alloc:
            alloc[c] = min(alloc[c], cats[c])
        diff = target_m - sum(alloc.values())
        order = sorted(cats, key=lambda c: -cats[c])
        j = 0
        while diff != 0 and j < 10 * len(order):
            c = order[j % len(order)]
            if diff > 0 and alloc[c] < cats[c]:
                alloc[c] += 1; diff -= 1
            elif diff < 0 and alloc[c] > 0:
                alloc[c] -= 1; diff += 1
            j += 1
        for c in alloc:                      # cells are paired internally: keep them even
            alloc[c] -= alloc[c] % 2
        for c, n in sorted(alloc.items()):
            cids = sorted(i for i in ids if i in A and A[i]["category_group"] == c)
            got = take(A, rng.sample(cids, min(n, len(cids))))
            rows += make_pairs(got, sub_rng(seed, f"M:{m}:{c}"), "M", m, pair_counter)

    # ---- D, D32, DB, F: category-proportional (floor 10%), month uniform ---
    def cat_month_draw(cohort: str, n_total: int, pool: dict[str, dict], rng: random.Random) -> list[dict]:
        cats = Counter(d["category_group"] for d in pool.values())
        tot = sum(cats.values())
        alloc = {c: max(math.ceil(0.10 * n_total), round(n_total * v / tot)) for c, v in cats.items()}
        # normalise to n_total
        while sum(alloc.values()) > n_total:
            c = max(alloc, key=lambda c: alloc[c]); alloc[c] -= 1
        got = []
        for c, n in sorted(alloc.items()):
            cids = sorted(i for i, d in pool.items() if d["category_group"] == c)
            # month-uniform inside the category
            by_m = defaultdict(list)
            for i in cids:
                by_m[pool[i]["month"]].append(i)
            ms = sorted(by_m)
            picks, k = [], 0
            order = [i for m in ms for i in rng.sample(sorted(by_m[m]), len(by_m[m]))]
            # round-robin over months
            queues = {m: iter(rng.sample(sorted(by_m[m]), len(by_m[m]))) for m in ms}
            while len(picks) < n and queues:
                for m in list(queues):
                    try:
                        picks.append(next(queues[m]))
                    except StopIteration:
                        del queues[m]
                    if len(picks) >= n:
                        break
            got += take(pool, picks)
        return got

    for cohort in ("D", "D32", "DB", "F"):
        rng = sub_rng(seed, cohort)
        pool_c = A
        if cohort == "F":
            pool_c = {i: d for i, d in A.items() if d["alt_text_form"] == "redpajama_latex"}
        got = cat_month_draw(cohort, 2 * SPEC[cohort]["per_arm"], pool_c, rng)
        if cohort == "F":
            for d in got:
                A.pop(d["doc_id"], None)
                attach_pair_score(d, f"F:{d['doc_id']}:inj", f"F:{d['doc_id']}:latex")
        prs = make_pairs(got, sub_rng(seed, f"{cohort}:pairs"), cohort, "all", pair_counter)
        if cohort == "D":
            # r per PAIR, block-randomized within bpb tertiles of the pair mean
            pairs = defaultdict(list)
            for r in prs:
                pairs[r["pair_id"]].append(r)
            pm = sorted(pairs, key=lambda p: (np.mean([x["bpb_pre"] for x in pairs[p]]), p))
            t = len(pm) // 3
            for ti, chunk in enumerate((pm[:t], pm[t:2 * t], pm[2 * t:])):
                levels = (D_LEVELS * math.ceil(len(chunk) / len(D_LEVELS)))[:len(chunk)]
                sub_rng(seed, f"D:r:{ti}").shuffle(levels)
                for p, r_val in zip(chunk, levels):
                    for x in pairs[p]:
                        x["r"] = r_val
                        x["stratum"] = f"r{r_val}"
        rows += prs

    # ---- X: shuffled sources (consume arXiv), model samples, random tokens --
    rng = sub_rng(seed, "X")
    xs_dir = out / "texts" / "X"
    xs_dir.mkdir(exist_ok=True)
    for stratum in ("sentence_shuffled", "word_shuffled"):
        src = take(A, rng.sample(sorted(A), X_SOURCE_DOCS))
        made = []
        for d in src:
            text = Path(d["text_path"]).read_text(encoding="utf-8")
            base = truncate_paragraph(text, tok, X_TARGET_TOKENS)
            shuf = sentence_shuffle(base, rng) if stratum == "sentence_shuffled" else word_shuffle(base, rng)
            doc_id = f"X:{stratum}:{d['doc_id']}"
            p = xs_dir / f"{safe(doc_id)}.txt"
            p.write_text(shuf, encoding="utf-8")
            made.append({**d, "doc_id": doc_id, "source_doc_id": d["doc_id"], "group": "anchor", "text_path": str(p),
                         "text_form": stratum, "alt_text_path": "", "alt_text_form": "",
                         "n_tokens": len(tok(shuf, add_special_tokens=False)["input_ids"]) + 1})
        rows += make_pairs(made, sub_rng(seed, f"X:{stratum}"), "X", stratum, pair_counter)
    # model samples (pre-generated)
    ms_csv = out / "texts" / "X_model_samples.csv.gz"
    if ms_csv.exists():
        made = []
        for r in rd(ms_csv):
            ids = np.load(r["ids_path"])
            made.append({"doc_id": r["doc_id"], "group": "anchor", "month": "", "category_group": "", "title": "",
                         "text_path": r["text_path"], "text_form": "model_samples", "ids_path": r["ids_path"],
                         "alt_text_path": "", "alt_text_form": "", "n_tokens": len(ids) + 1, "bpb_pre": float("nan")})
        rows += make_pairs(made, sub_rng(seed, "X:model_samples"), "X", "model_samples", pair_counter)
    else:
        log("WARNING: no model samples (run gen-samples first); X:model_samples skipped")
    # random tokens
    vocab_ids = [i for i in range(len(tok)) if i not in set(tok.all_special_ids)]
    made = []
    rr = np.random.default_rng(int(sub_rng(seed, "X:random").random() * 2**31))
    for k in range(2 * SPEC["X"]["per_arm"]):
        ids = rr.choice(vocab_ids, size=X_TARGET_TOKENS).astype(np.int32)
        doc_id = f"X:random_tokens:{k:03d}"
        p = xs_dir / f"{safe(doc_id)}.ids.npy"
        np.save(p, ids)
        (xs_dir / f"{safe(doc_id)}.txt").write_text(tok.decode(ids.tolist()), encoding="utf-8")
        made.append({"doc_id": doc_id, "group": "anchor", "month": "", "category_group": "", "title": "",
                     "text_path": str(xs_dir / f"{safe(doc_id)}.txt"), "text_form": "random_tokens", "ids_path": str(p),
                     "alt_text_path": "", "alt_text_form": "", "n_tokens": X_TARGET_TOKENS + 1, "bpb_pre": float("nan")})
    rows += make_pairs(made, sub_rng(seed, "X:random_tokens"), "X", "random_tokens", pair_counter)

    # ---- W-new, W-gap, S-se: month-uniform ----------------------------------
    for cohort, src in (("W-new", "wikiA"), ("W-gap", "wikiB"), ("S-se", "se")):
        rng = sub_rng(seed, cohort)
        pool_c = pools[src]
        n_total = 2 * SPEC[cohort]["per_arm"]
        months = sorted({d["month"] for d in pool_c.values()})
        by_m = {m: sorted(i for i, d in pool_c.items() if d["month"] == m) for m in months}
        per, remaining, open_m = {m: 0 for m in months}, n_total, set(months)
        while remaining > 0 and open_m:
            share = max(1, remaining // len(open_m))
            for m in sorted(open_m):
                add = min(share, len(by_m[m]) - per[m], remaining)
                per[m] += add; remaining -= add
                if per[m] >= len(by_m[m]):
                    open_m.discard(m)
                if remaining <= 0:
                    break
        got = []
        for m in months:
            got += take(pool_c, rng.sample(by_m[m], per[m]))
        rows += make_pairs(got, sub_rng(seed, f"{cohort}:pairs"), cohort, "all", pair_counter)

    # ---- E0, E0-D (identical), E (five strata) -------------------------------
    Ep = pools["edited"]
    ident = sorted(i for i, d in Ep.items() if d["novelty_stratum"] == "identical")
    rng = sub_rng(seed, "E0")
    got = take(Ep, rng.sample(ident, 2 * SPEC["E0"]["per_arm"]))
    rows += make_pairs(got, sub_rng(seed, "E0:pairs"), "E0", "identical", pair_counter)
    ident = sorted(i for i, d in Ep.items() if d["novelty_stratum"] == "identical")
    got = take(Ep, sub_rng(seed, "E0-D").sample(ident, 2 * SPEC["E0-D"]["per_arm"]))
    rows += make_pairs(got, sub_rng(seed, "E0-D:pairs"), "E0-D", "identical", pair_counter)
    for s in E_STRATA:
        ids = sorted(i for i, d in Ep.items() if d["novelty_stratum"] == s)
        got = take(Ep, sub_rng(seed, f"E:{s}").sample(ids, min(2 * SPEC["E"]["per_arm"], len(ids))))
        rows += make_pairs(got, sub_rng(seed, f"E:{s}:pairs"), "E", s, pair_counter)

    # ---- null pools ---------------------------------------------------------
    null_rows = []
    for src, n in NULL.items():
        pool_c = pools[src] if src != "arxiv" else A
        ids = sorted(pool_c)
        got = take(pool_c, sub_rng(seed, f"N:{src}").sample(ids, min(n, len(ids))))
        for d in got:
            null_rows.append({**d, "cohort": f"N-{src}", "arm": "null"})
    log("left over: " + ", ".join(f"{k} {len(v):,}" for k, v in sorted(pools.items())))

    # ---- manipulations, texts, splits ---------------------------------------
    for r in rows:
        r.setdefault("r", SPEC[r["cohort"]].get("r", 1))
        r["burst"] = int(SPEC[r["cohort"]].get("burst", False))
        r.setdefault("target_len", "")
        r["inject_text_path"] = r["text_path"]
        r["inject_text_form"] = r["text_form"]
    l_dir = out / "texts" / "L"
    l_dir.mkdir(exist_ok=True)
    for r in rows:
        if r["cohort"] == "L":
            text = Path(r["text_path"]).read_text(encoding="utf-8")
            trunc = truncate_paragraph(text, tok, int(r["target_len"]))
            p = l_dir / f"{safe(r['doc_id'])}.L{r['target_len']}.txt"
            p.write_text(trunc, encoding="utf-8")
            r["inject_text_path"] = str(p)
        if r["cohort"] == "F":
            # injected arm gets the LaTeX-derived text; the holdout partner is
            # scored in both forms, so it keeps its PDF text and the alt path
            if r["arm"] == "inject":
                r["inject_text_path"], r["inject_text_form"] = r["alt_text_path"], r["alt_text_form"]
    # dev / holdout split by pair
    pids = sorted({r["pair_id"] for r in rows})
    rng = sub_rng(seed, "split")
    dev = set(rng.sample(pids, int(round(DEV_FRAC * len(pids)))))
    for r in rows:
        r["split"] = "dev" if r["pair_id"] in dev else "holdout"
    rng = sub_rng(seed, "split:null")
    for r in null_rows:
        r["split"] = "dev" if rng.random() < DEV_FRAC else "holdout"

    # ---- tokenize (the exact text that is injected / evaluated) --------------
    tok_dir = out / "tokens"
    for r in rows:
        d = tok_dir / r["cohort"]
        d.mkdir(parents=True, exist_ok=True)
        if r.get("ids_path"):
            ids = np.load(r["ids_path"]).astype(np.int32)
        else:
            text = Path(r["inject_text_path"]).read_text(encoding="utf-8")
            ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int32)
        ids = np.concatenate([ids, np.array([eos], dtype=np.int32)])
        np.save(d / f"{safe(r['doc_id'])}.npy", ids)
        r["n_tokens_inject"] = int(len(ids))
        r["tokens_path"] = str(d / f"{safe(r['doc_id'])}.npy")
    log(f"tokenized {len(rows):,} documents")

    # ---- copies and positions ------------------------------------------------
    copies = []
    for cohort in sorted({r["cohort"] for r in rows}):
        inj = [r for r in rows if r["cohort"] == cohort and r["arm"] == "inject"]
        rng = sub_rng(seed, f"pos:{cohort}")
        if SPEC[cohort].get("burst"):
            anchors = [(k % 10) for k in range(len(inj))]
            rng.shuffle(anchors)
            for r, dec in zip(inj, anchors):
                lo = TRIM + dec * (1 - 2 * TRIM) / 10
                anchor = lo + rng.random() * ((1 - 2 * TRIM) / 10 - BURST_WINDOW)
                for c in range(int(r["r"])):
                    copies.append({"doc_id": r["doc_id"], "cohort": cohort, "copy": c, "decile": dec,
                                   "position_frac": round(anchor + rng.random() * BURST_WINDOW, 8), "burst": 1})
            continue
        units = [(r, c) for r in inj for c in range(int(r["r"]))]
        rng.shuffle(units)
        for k, (r, c) in enumerate(units):
            dec = k % 10
            lo = TRIM + dec * (1 - 2 * TRIM) / 10
            copies.append({"doc_id": r["doc_id"], "cohort": cohort, "copy": c, "decile": dec,
                           "position_frac": round(lo + rng.random() * (1 - 2 * TRIM) / 10, 8), "burst": 0})
    copies.sort(key=lambda c: (c["position_frac"], c["doc_id"]))

    # ---- write ----------------------------------------------------------------
    cols = ["doc_id", "cohort", "stratum", "arm", "pair_id", "split", "r", "burst", "target_len", "group", "month",
            "category_group", "title", "n_tokens", "n_tokens_inject", "bpb_pre", "length_bin", "text_path", "text_form",
            "inject_text_path", "inject_text_form", "alt_text_path", "alt_text_form", "tokens_path",
            "novelty_stratum", "frac_novel_tokens", "mask_path", "ancestor_record_id", "source_doc_id",
            "pair_bpb", "pair_n_tokens"]
    for r in rows:
        for c in cols:
            r.setdefault(c, "")
    wr(out / "manifest.csv.gz", rows, cols)
    wr(out / "copies.csv.gz", copies)
    ncols = ["doc_id", "cohort", "arm", "split", "group", "month", "category_group", "title", "n_tokens", "bpb_pre",
             "text_path", "text_form", "alt_text_path", "alt_text_form", "novelty_stratum", "mask_path"]
    for r in null_rows:
        for c in ncols:
            r.setdefault(c, "")
    wr(out / "null_pool.csv.gz", null_rows, ncols)

    # ---- summary ---------------------------------------------------------------
    summ = {"seed": seed, "model": args.model, "tokenization": EOS_NOTE, "stage2_tokens": args.stage2_tokens,
            "filters": {"arxiv_tokens": [ARXIV_MIN, ARXIV_MAX], "short_domains_min_tokens": SHORT_MIN},
            "cohorts": {}, "null_pools": dict(Counter(r["cohort"] for r in null_rows))}
    tot_inj = tot_docs = 0
    for cohort in SPEC:
        cr = [r for r in rows if r["cohort"] == cohort]
        if not cr:
            continue
        inj = [r for r in cr if r["arm"] == "inject"]
        toks = sum(int(r["r"]) * r["n_tokens_inject"] for r in inj)
        tot_inj += toks
        tot_docs += len(inj)
        summ["cohorts"][cohort] = {
            "question": SPEC[cohort]["question"], "pairs": len(inj), "per_arm": len(inj),
            "strata": dict(sorted(Counter(r["stratum"] for r in inj).items())),
            "copies": sum(int(r["r"]) for r in inj), "tokens_injected": toks,
            "share_of_stage2": toks / args.stage2_tokens,
            "median_n_tokens_inject": int(np.median([r["n_tokens_inject"] for r in inj])),
        }
    summ["total"] = {"pairs": tot_docs, "documents_both_arms": len(rows), "copies": len(copies),
                     "tokens_injected": tot_inj, "share_of_stage2": tot_inj / args.stage2_tokens,
                     "null_documents": len(null_rows)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"\n{'cohort':7}{'pairs':>6}{'copies':>7}{'tokens':>12}{'share':>9}")
    for c, v in summ["cohorts"].items():
        log(f"{c:7}{v['pairs']:6,}{v['copies']:7,}{v['tokens_injected']:12,}{100 * v['share_of_stage2']:8.3f}%")
    t = summ["total"]
    log(f"{'TOTAL':7}{t['pairs']:6,}{t['copies']:7,}{t['tokens_injected']:12,}{100 * t['share_of_stage2']:8.3f}%   "
        f"null {t['null_documents']:,}")
    log(f"wrote {out}/manifest.csv.gz, copies.csv.gz, null_pool.csv.gz, summary.json, tokens/")
    return 0




# ---------------------------------------------------------------------------
# refill
# ---------------------------------------------------------------------------
def cmd_refill(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos = tok.eos_token_id
    out = args.out_dir
    seed = args.seed
    man = rd(out / "manifest.csv.gz")
    copies = rd(out / "copies.csv.gz")
    null = rd(out / "null_pool.csv.gz")
    cols = list(man[0].keys())
    excl = {r["doc_id"] for r in rd(args.exclusions)}
    clean = {r["doc_id"] for r in rd(args.clean)}          # documents that passed the FULL audit
    clean |= {r["doc_id"] for r in rd(args.edited_clean)} if args.edited_clean else set()

    # ---- which pairs go ------------------------------------------------------
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for r in man:
        by_pair[r["pair_id"]].append(r)
    dropped_pairs = sorted({r["pair_id"] for r in man
                            if r["doc_id"] in excl or (r.get("source_doc_id") and r["source_doc_id"] in excl)})
    if args.restore_unfilled:
        # slots that an earlier refill could not fill: present in that run's
        # backup manifest, absent now. Their template rows and copy positions
        # come from the backup.
        b_man = rd(args.restore_unfilled)
        b_cop = rd(Path(str(args.restore_unfilled).replace("manifest_before_", "copies_before_")))
        gone = sorted({r["pair_id"] for r in b_man} - set(by_pair))
        for r in b_man:
            if r["pair_id"] in gone:
                by_pair[r["pair_id"]].append(r)
        copies = copies + [c for c in b_cop if any(c["doc_id"] == r["doc_id"] and r["arm"] == "inject"
                                                   for r in b_man if r["pair_id"] in gone)]
        dropped_pairs = sorted(set(dropped_pairs) | set(gone))
        log(f"restoring {len(gone)} unfilled slots from {args.restore_unfilled.name}: "
            f"{dict(Counter(by_pair[p][0]['cohort'] for p in gone))}")
    log(f"exclusions {len(excl):,}; pairs to replace {len(dropped_pairs):,} "
        f"{dict(Counter(by_pair[p][0]['cohort'] for p in dropped_pairs))}")
    used = {r["doc_id"] for r in man} | {r["doc_id"] for r in null}
    used |= {r["source_doc_id"] for r in man if r.get("source_doc_id")}   # arXiv papers behind the shuffled anchors

    # ---- spare pools (same eligibility rules as build) ------------------------
    sc = {r["doc_id"]: r for r in rd(args.audit_dir / "logprobs_stage1" / "scores.csv.gz") if r["status"] == "ok"}
    arx = {r["doc_id"]: r for r in rd(args.arxiv_pool)}
    spare: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rd(args.audit_dir / "stageG" / "scoring_manifest.csv.gz"):
        s_ = sc.get(r["doc_id"])
        if not s_ or r["doc_id"] in used or r["doc_id"] not in clean or r["doc_id"] in excl:
            continue
        d = {"doc_id": r["doc_id"], "group": r["group"], "month": r["month"], "category_group": r["category_group"],
             "title": r["title"], "text_path": r["raw_text_path"], "text_form": r["text_form"],
             "n_tokens": int(s_["n_tokens"]), "bpb_pre": float(s_["bpb"]), "alt_text_path": "", "alt_text_form": ""}
        if r["group"] == "postcutoff":
            if not (ARXIV_MIN <= d["n_tokens"] <= ARXIV_MAX):
                continue
            a = arx.get(r["doc_id"], {})
            if r["text_form"] == "pes2o_style_pdf" and a.get("latex_raw_status") == "ok":
                d["alt_text_path"], d["alt_text_form"] = a["latex_raw_path"], "redpajama_latex"
            spare["arxiv"][d["doc_id"]] = d
        elif d["n_tokens"] >= SHORT_MIN:
            spare[r["group"]][d["doc_id"]] = d
    ed_sc = {r["doc_id"]: r for r in rd(args.edited_dir / "logprobs_stage1" / "scores.csv.gz") if r["status"] == "ok"}
    for r in rd(args.edited_dir / "edited_group.csv.gz"):
        s_ = ed_sc.get(r["doc_id"])
        if not s_ or int(s_["n_tokens"]) < SHORT_MIN or r["doc_id"] in used or r["doc_id"] in excl:
            continue
        spare["edited"][r["doc_id"]] = {
            "doc_id": r["doc_id"], "group": "edited", "month": r["last_revision"][:7], "category_group": "",
            "title": r["title"], "text_path": r["raw_text_path"], "text_form": "dolma_wiki",
            "n_tokens": int(s_["n_tokens"]), "bpb_pre": float(s_["bpb"]), "alt_text_path": "", "alt_text_form": "",
            "novelty_stratum": "identical" if int(r["n_words_novel"]) == 0 else r["stratum"],
            "frac_novel_tokens": float(r["frac_novel_tokens"]), "mask_path": r["mask_path"],
            "ancestor_record_id": r["ancestor_record_id"]}
    # null pool as reserve (only rows that are still clean)
    reserve: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in null:
        src = r["cohort"].replace("N-", "")
        if r["doc_id"] in excl or (src != "edited" and r["doc_id"] not in clean):
            continue
        d = dict(r); d["n_tokens"] = int(r["n_tokens"]); d["bpb_pre"] = float(r["bpb_pre"])
        reserve[src][r["doc_id"]] = d
    log("spares: " + ", ".join(f"{k} {len(v):,}" for k, v in sorted(spare.items()))
        + " | reserve (null pool): " + ", ".join(f"{k} {len(v):,}" for k, v in sorted(reserve.items())))

    SRC = {"M": "arxiv", "D": "arxiv", "D32": "arxiv", "DB": "arxiv", "L": "arxiv", "F": "arxiv",
           "W-new": "wikiA", "W-gap": "wikiB", "S-se": "se", "E0": "edited", "E0-D": "edited", "E": "edited"}

    def adjacent_months(m: str) -> set[str]:
        y, mo = int(m[:4]), int(m[5:7])
        prev = f"{y - 1}-12" if mo == 1 else f"{y}-{mo - 1:02d}"
        nxt = f"{y + 1}-01" if mo == 12 else f"{y}-{mo + 1:02d}"
        return {prev, nxt}

    def eligible(cohort: str, stratum: str, tmpl: dict, pool: dict, relax: int = 0) -> list[str]:
        """relax 0: exact stratum cell; 1 (M only): same month, any category;
        2 (M only): adjacent month, same category (2025-04 is a partial month)."""
        ids = []
        for i, d in pool.items():
            if cohort == "M":
                if relax < 2 and d["month"] != stratum:
                    continue
                if relax == 2 and (d["month"] not in adjacent_months(stratum) or d["category_group"] != tmpl["category_group"]):
                    continue
                if relax == 0 and d["category_group"] != tmpl["category_group"]:
                    continue
            if cohort == "L" and d["n_tokens"] < 8192:
                continue
            if cohort == "F" and d.get("alt_text_form") != "redpajama_latex":
                continue
            if cohort in ("E0", "E0-D") and d.get("novelty_stratum") != "identical":
                continue
            if cohort == "E" and d.get("novelty_stratum") != stratum:
                continue
            ids.append(i)
        return sorted(ids)

    new_rows, new_copies, taken_from_reserve, unfilled = [], [], Counter(), []
    keep_copies = [c for c in copies]
    for pid in dropped_pairs:
        old = by_pair[pid]
        cohort, stratum = old[0]["cohort"], old[0]["stratum"]
        rng = sub_rng(seed, f"refill:{pid}")
        old_inj_id = next(x["doc_id"] for x in old if x["arm"] == "inject")
        keep_copies = [c for c in keep_copies if c["doc_id"] != old_inj_id]   # dropped either way
        if cohort == "X":
            if stratum not in ("sentence_shuffled", "word_shuffled"):
                unfilled.append((pid, cohort, f"{stratum} anchors have no source to replace"))
                continue
            cand = sorted(spare["arxiv"])
            if len(cand) < 2:
                unfilled.append((pid, cohort, "no spare arXiv source")); continue
            xs_dir = out / "texts" / "X"
            picks = [spare["arxiv"].pop(i) for i in rng.sample(cand, 2)]
            if rng.random() < 0.5:
                picks = picks[::-1]
            old_inj = next(x for x in old if x["arm"] == "inject")
            for d, arm in zip(picks, ("inject", "holdout")):
                text = Path(d["text_path"]).read_text(encoding="utf-8")
                base = truncate_paragraph(text, tok, X_TARGET_TOKENS)
                shuf = sentence_shuffle(base, rng) if stratum == "sentence_shuffled" else word_shuffle(base, rng)
                doc_id = f"X:{stratum}:{d['doc_id']}"
                ptxt = xs_dir / f"{safe(doc_id)}.txt"
                ptxt.write_text(shuf, encoding="utf-8")
                ids = np.asarray(tok(shuf, add_special_tokens=False)["input_ids"], dtype=np.int32)
                ids = np.concatenate([ids, np.array([eos], dtype=np.int32)])
                tp = out / "tokens" / "X" / f"{safe(doc_id)}.npy"
                np.save(tp, ids)
                r = {c: "" for c in cols}
                r.update({"doc_id": doc_id, "source_doc_id": d["doc_id"], "group": "anchor", "month": d["month"],
                          "category_group": d["category_group"], "title": d["title"], "n_tokens": len(ids),
                          "bpb_pre": d["bpb_pre"], "text_path": str(ptxt), "text_form": stratum,
                          "inject_text_path": str(ptxt), "inject_text_form": stratum, "tokens_path": str(tp),
                          "n_tokens_inject": int(len(ids)), "cohort": cohort, "stratum": stratum, "arm": arm,
                          "pair_id": pid, "split": old[0]["split"], "r": old[0]["r"], "burst": old[0]["burst"],
                          "target_len": "", "replaced": "1", "from_null_reserve": "",
                          "replaces": next(x["doc_id"] for x in old if x["arm"] == arm)})
                new_rows.append(r)
                if arm == "inject":
                    for c in copies:
                        if c["doc_id"] == old_inj["doc_id"]:
                            new_copies.append({**c, "doc_id": doc_id})
            continue
        src = SRC[cohort]
        # ladder: spares exact cell -> null reserve exact cell -> (M) spares same month -> (M) reserve same month
        ladder = [(spare[src], 0, False), (reserve[src], 0, True)]
        if cohort == "M":
            ladder += [(spare[src], 1, False), (reserve[src], 1, True), (spare[src], 2, False), (reserve[src], 2, True)]
        cand, pool, from_reserve = [], None, False
        for pool_, relax, res in ladder:
            cand = eligible(cohort, stratum, old[0], pool_, relax)
            if len(cand) >= 2:
                pool, from_reserve = pool_, res
                if relax == 1:
                    log(f"  {pid}: no spare in {stratum}/{old[0]['category_group']}; using same month, any category")
                if relax == 2:
                    log(f"  {pid}: month {stratum} exhausted; using an adjacent month, same category")
                break
        if pool is None:
            unfilled.append((pid, cohort, f"no spare in {src}/{stratum}"))
            continue
        # draw 2 nearest in bpb to the dropped pair's mean (keeps the stratum's balance), tie-broken at random
        target = np.mean([float(x["bpb_pre"]) for x in old])
        cand.sort(key=lambda i: (abs(pool[i]["bpb_pre"] - target), rng.random()))
        pick = [pool.pop(i) for i in cand[:2]]
        if from_reserve:
            taken_from_reserve[src] += 2
        if rng.random() < 0.5:
            pick = pick[::-1]
        old_inj = next(x for x in old if x["arm"] == "inject")
        for d, arm in zip(pick, ("inject", "holdout")):
            r = {c: "" for c in cols}
            r.update({k: v for k, v in d.items() if k in cols})
            r.update({"cohort": cohort, "stratum": stratum, "arm": arm, "pair_id": pid, "split": old[0]["split"],
                      "r": old[0]["r"], "burst": old[0]["burst"], "target_len": old[0]["target_len"],
                      "replaced": "1", "replaces": next(x["doc_id"] for x in old if x["arm"] == arm),
                      "from_null_reserve": "1" if from_reserve else ""})
            r["inject_text_path"], r["inject_text_form"] = r["text_path"], r["text_form"]
            if cohort == "L":
                text = Path(r["text_path"]).read_text(encoding="utf-8")
                p = out / "texts" / "L" / f"{safe(r['doc_id'])}.L{r['target_len']}.txt"
                p.write_text(truncate_paragraph(text, tok, int(r["target_len"])), encoding="utf-8")
                r["inject_text_path"] = str(p)
            if cohort == "F" and arm == "inject":
                r["inject_text_path"], r["inject_text_form"] = r["alt_text_path"], r["alt_text_form"]
            ids = np.asarray(tok(Path(r["inject_text_path"]).read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"], dtype=np.int32)
            ids = np.concatenate([ids, np.array([eos], dtype=np.int32)])
            tp = out / "tokens" / cohort / f"{safe(r['doc_id'])}.npy"
            np.save(tp, ids)
            r["n_tokens_inject"], r["tokens_path"] = int(len(ids)), str(tp)
            new_rows.append(r)
            if arm == "inject":       # inherit the dropped inject document's copy positions
                for c in copies:
                    if c["doc_id"] == old_inj["doc_id"]:
                        new_copies.append({**c, "doc_id": r["doc_id"]})

    # ---- write ---------------------------------------------------------------
    dropped_ids = {x["doc_id"] for p in dropped_pairs for x in by_pair[p]}
    kept = [r for r in man if r["doc_id"] not in dropped_ids]
    for r in kept:
        r.setdefault("replaced", ""); r.setdefault("replaces", ""); r.setdefault("from_null_reserve", "")
    cols2 = cols + [c for c in ("replaced", "replaces", "from_null_reserve") if c not in cols]
    ver = args.version_tag
    wr(out / f"manifest_before_{ver}.csv.gz", man, cols)
    wr(out / f"copies_before_{ver}.csv.gz", copies)
    wr(out / f"null_pool_before_{ver}.csv.gz", null)
    wr(out / "manifest.csv.gz", kept + new_rows, cols2)
    wr(out / "copies.csv.gz", sorted(keep_copies + new_copies, key=lambda c: (float(c["position_frac"]), c["doc_id"])))
    used_reserve = {r["doc_id"] for r in new_rows if r["from_null_reserve"]}
    null_kept = [r for r in null if r["doc_id"] not in excl and r["doc_id"] not in used_reserve]
    wr(out / "null_pool.csv.gz", null_kept)
    wr(out / f"refill_{ver}.csv.gz", [{"pair_id": r["pair_id"], "cohort": r["cohort"], "stratum": r["stratum"], "arm": r["arm"],
                                       "new_doc_id": r["doc_id"], "replaces": r["replaces"], "from_null_reserve": r["from_null_reserve"]}
                                      for r in new_rows] or [{"pair_id": "", "cohort": "", "stratum": "", "arm": "", "new_doc_id": "", "replaces": "", "from_null_reserve": ""}])
    log(f"replaced {len(new_rows) // 2:,} pairs; unfilled {len(unfilled)}; from null reserve {dict(taken_from_reserve)}; "
        f"null pool {len(null):,} -> {len(null_kept):,} (excluded {sum(1 for r in null if r['doc_id'] in excl)}, used as reserve {len(used_reserve)})")
    for pid, cohort, why in unfilled:
        log(f"  UNFILLED {pid} ({cohort}): {why}")
    summ = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    summ.setdefault("refills", []).append({"version": ver, "exclusions": len(excl), "pairs_replaced": len(new_rows) // 2,
                                           "unfilled": [list(u) for u in unfilled], "from_null_reserve": dict(taken_from_reserve),
                                           "null_pool_after": len(null_kept)})
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"manifest/copies/null_pool rewritten; previous versions kept as *_before_{ver}.csv.gz")
    return 0 if not unfilled else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=paths.BUNDLE)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen-samples")
    g.add_argument("--revision", default="stage1-step1907359-tokens4001B")
    g.set_defaults(fn=cmd_gen_samples)
    b = sub.add_parser("build")
    b.add_argument("--audit-dir", type=Path, default=W / "audit_v2")
    b.add_argument("--edited-dir", type=Path, default=W / "wiki_edited")
    b.add_argument("--arxiv-pool", type=Path, default=W / "postcutoff" / "pool_parsed_merged.csv.gz")
    b.add_argument("--pair-scores", type=Path, default=None,
                   help="scores.csv.gz of the as-injected texts (doc ids L:<doc>:inj, F:<doc>:inj / F:<doc>:latex) "
                        "from a previous build's logprobs_extra_stage1; L and F are then paired on the prefix / "
                        "LaTeX form instead of the source document")
    b.add_argument("--stage2-tokens", type=float, default=50e9,
                   help="tokens in the stage-2 run the injection is measured against (dolmino 50B)")
    b.set_defaults(fn=cmd_build)
    f = sub.add_parser("refill")
    f.add_argument("--exclusions", type=Path, required=True, help="csv.gz with a doc_id column: documents to remove")
    f.add_argument("--clean", type=Path, required=True, help="candidates_clean.csv.gz of the FULL audit (all targets)")
    f.add_argument("--edited-clean", type=Path, default=None, help="optional doc_id list for edited pages still usable")
    f.add_argument("--audit-dir", type=Path, default=W / "audit_v2")
    f.add_argument("--edited-dir", type=Path, default=W / "wiki_edited")
    f.add_argument("--arxiv-pool", type=Path, default=W / "postcutoff" / "pool_parsed_merged.csv.gz")
    f.add_argument("--version-tag", default="refill1")
    f.add_argument("--restore-unfilled", type=Path, default=None,
                   help="manifest_before_<tag>.csv.gz of an earlier refill whose unfilled slots should be filled now "
                        "(e.g. from documents freed as pair partners)")
    f.set_defaults(fn=cmd_refill)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
