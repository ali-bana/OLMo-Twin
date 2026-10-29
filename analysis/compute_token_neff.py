#!/usr/bin/env python3
"""Per-token effective-sample-size statistics for the injection-v2 membership signal.

For every scored document we load the per-token log-probabilities under the
treated model (`injected`) and the control (`seed42`), form the per-token oracle
delta  d_t = logp_inj(t) - logp_ctl(t), and record

  * the delta mean / sd / token count,
  * the integrated autocorrelation time tau of the delta series (Sokal window),
    giving an effective number of independent tokens n_eff = n_tok / tau,
  * the running document score restricted to the FIRST T tokens for a ladder of
    token budgets T -- this lets the figure code recompute the within-pair
    effect size d(T) as a function of how much of each document the auditor is
    allowed to read.

Inputs (under the data root, see olmo_twin/paths.py): the per-token files
run/scores/{injected,official-seed42}-step23852/tokens/*.npz, which must still
contain token_ids (a document whose files cannot be read, or lack token_ids,
gets NaN statistics), and analysis/per_document_attacks_prelim.csv.gz (document
order).
Output: analysis/token_neff.npz, one .npz with parallel arrays indexed the same
way as `doc_ids`. Pure CPU + IO; NWORKERS sets the number of processes
(default 32).
"""
from __future__ import annotations

import gzip
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

SCORES = str(paths.RUN / "scores")
INJ_DIR = f"{SCORES}/injected-step23852/tokens"
CTL_DIR = f"{SCORES}/official-seed42-step23852/tokens"
TABLE = str(paths.ANALYSIS / "per_document_attacks_prelim.csv.gz")
OUT = str(paths.ANALYSIS / "token_neff.npz")

BUDGETS = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768], dtype=np.int64)
MAXLAG = 256


def safe(doc_id: str) -> str:
    return doc_id.replace("/", "_").replace(":", "_")


def integrated_act(x: np.ndarray, maxlag: int = MAXLAG) -> float:
    """Sokal-windowed integrated autocorrelation time of a 1-D series."""
    n = x.size
    if n < 32:
        return 1.0
    x = x - x.mean()
    v0 = float(np.dot(x, x) / n)
    if v0 <= 0:
        return 1.0
    m = min(maxlag, n // 4)
    # FFT autocovariance
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[: m + 1] / n
    rho = acov / acov[0]
    tau = 1.0
    for k in range(1, m + 1):
        tau += 2.0 * rho[k]
        if k >= 6.0 * tau:          # Sokal automatic window
            break
    return float(max(tau, 1.0))


def one(doc_id: str):
    fi = os.path.join(INJ_DIR, safe(doc_id) + ".npz")
    fc = os.path.join(CTL_DIR, safe(doc_id) + ".npz")
    nb = len(BUDGETS)
    bad = (0, np.nan, np.nan, np.nan, np.nan, np.full(nb, np.nan), np.full(nb, np.nan), np.nan)
    if not (os.path.exists(fi) and os.path.exists(fc)):
        return bad
    try:
        zi = np.load(fi)
        zc = np.load(fc)
        li = zi["logprobs"].astype(np.float64)
        lc = zc["logprobs"].astype(np.float64)
        ti = zi["token_ids"]
        tc = zc["token_ids"]
    except Exception:
        return bad
    n = min(li.size, lc.size)
    if n == 0:
        return bad
    li, lc, ti, tc = li[:n], lc[:n], ti[:n], tc[:n]
    aligned = bool(np.array_equal(ti, tc))
    ok = np.isfinite(li) & np.isfinite(lc)
    if ok.sum() < 8:
        return bad
    delta = (li - lc)[ok]
    lci = lc[ok]
    nt = delta.size
    # cumulative prefix means at the budget ladder (positions counted in SCORED tokens)
    cs = np.cumsum(delta)
    csc = np.cumsum(lci)
    pref = np.full(nb, np.nan)
    prefc = np.full(nb, np.nan)
    for j, T in enumerate(BUDGETS):
        if nt >= T:
            pref[j] = cs[T - 1] / T
            prefc[j] = csc[T - 1] / T
    tau = integrated_act(delta)
    return (nt, float(delta.mean()), float(delta.std(ddof=1)) if nt > 1 else np.nan,
            tau, float(nt / tau), pref, prefc, float(aligned))


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(TABLE, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    doc_ids = [r["doc_id"] for r in rows]
    print(f"{len(doc_ids)} documents", flush=True)
    t0 = time.time()
    nw = int(os.environ.get("NWORKERS", "32"))
    with Pool(nw) as pool:
        res = pool.map(one, doc_ids, chunksize=64)
    print(f"scored in {time.time() - t0:.1f}s with {nw} workers", flush=True)

    nb = len(BUDGETS)
    n_tok = np.array([r[0] for r in res], dtype=np.int64)
    dmean = np.array([r[1] for r in res])
    dsd = np.array([r[2] for r in res])
    tau = np.array([r[3] for r in res])
    neff = np.array([r[4] for r in res])
    pref = np.vstack([r[5] for r in res]) if res else np.zeros((0, nb))
    prefc = np.vstack([r[6] for r in res]) if res else np.zeros((0, nb))
    aligned = np.array([r[7] for r in res])

    np.savez_compressed(
        OUT,
        doc_ids=np.array(doc_ids, dtype=object),
        budgets=BUDGETS,
        n_tok=n_tok,
        delta_mean=dmean,
        delta_sd=dsd,
        tau=tau,
        n_eff=neff,
        prefix_delta=pref,
        prefix_ctl_logp=prefc,
        token_ids_aligned=aligned,
    )
    good = np.isfinite(tau)
    print(f"wrote {OUT}")
    print(f"  ok={good.sum()}  token-id alignment={np.nanmean(aligned):.4f}")
    print(f"  median n_tok={np.median(n_tok[good]):.0f} median tau={np.median(tau[good]):.2f} "
          f"median n_eff={np.median(neff[good]):.0f}")
    print(f"  median tokens-per-independent-observation={np.median(tau[good]):.2f}")


if __name__ == "__main__":
    main()
