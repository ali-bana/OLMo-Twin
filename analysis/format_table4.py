#!/usr/bin/env python3
"""Format the recomputed (all-NAT1) values the way paper_figures.calibrated() formats them,
and write (1) proposed macro lines, (2) a Table 4 variant whose CIs reuse the paper's existing
whole-NAT1 CI macros where the prose already prints a CI for the same statistic.

Reads values.json and calibrated_allNAT1.tex from --in (the output directory of
analysis/recompute_table4_all_pairs.py) and the whole-NAT1 CI macros from
--numbers (default: tables/numbers.tex in the default output directory of
figures/paper_figures.py; the paper's reformatted copy gives the same strings).
Writes macros_allNAT1.tex, macros_allNAT1_formatted.json and
calibrated_allNAT1_textCIs.tex next to the inputs.
"""
import argparse, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--in", dest="indir", type=Path, default=paths.ANALYSIS / "table4_all_pairs",
                 help="directory written by analysis/recompute_table4_all_pairs.py (default: analysis/table4_all_pairs)")
_ap.add_argument("--numbers", type=Path, default=paths.PAPER_OUT / "tables/numbers.tex",
                 help="numbers.tex with the whole-NAT1 CI macros (default: paper/tables/numbers.tex under the data root)")
ARGS = _ap.parse_args()
HERE = ARGS.indir                     # inputs and outputs
V = json.load(open(HERE / "values.json"))
NUM = ARGS.numbers
# the paper's copy holds bare values; the generator's own numbers.tex adds \xspace and {,}
_num = NUM.read_text().replace("{,}", "").replace("\\xspace", "")
pm = {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", _num)}
fmt = {}
for k, v in V.items():
    if not k.startswith("macro."):
        continue
    name = k[6:]
    if name in ("calibn", "calibnholdout", "calibkneg", "calibdraws"):
        s = str(int(v))
    elif name == "calibtaumedian":
        s = f"{v:.2f}"
    elif name == "calibneffshare":
        s = f"{v:.0f}"
    elif re.search(r"d(auc|auclo|auchi)$", name):
        s = f"{v:+.3f}"
    elif re.search(r"d(tpr|tprlo|tprhi|strat)$", name):
        s = f"{v:+.1f}"
    elif re.search(r"(auc|auclo|auchi)$", name):
        s = f"{v:.3f}"
    else:
        s = f"{v:.1f}"
    fmt[name] = s
used = ["calibn", "calibnholdout", "calibkneg", "calibratioorctpr", "calibzorctpr", "calibzorcdtpr", "calibzorcdtprlo",
        "calibzorcdtprhi", "calibzorcdauc", "calibzorcdauclo", "calibzorcdauchi", "calibratiosibtpr", "calibzsibtpr",
        "calibzsibdauc", "calibtaumedian", "calibneffshare", "calibzTorcauc", "calibzorcauc", "calibzTorcstrat", "calibzorcstrat"]
L = ["% proposed: holdout-dependent macros recomputed on all NAT1 pairs (analysis/recompute_table4_all_pairs.py); old value in the comment"]
for n in used:
    L.append(f"\\newcommand{{\\{n}}}{{{fmt[n]}}}  % was {pm.get(n, '?')}")
L.append("% other calib* macros (not cited by the current text), recomputed on all NAT1 pairs")
for n in sorted(fmt):
    if n not in used:
        L.append(f"\\newcommand{{\\{n}}}{{{fmt[n]}}}  % was {pm.get(n, '(none)')}")
(HERE / "macros_allNAT1.tex").write_text("\n".join(L) + "\n")
json.dump(fmt, open(HERE / "macros_allNAT1_formatted.json", "w"), indent=1, sort_keys=True)

# Table 4 variant: CIs from the paper's existing whole-NAT1 macros where they exist (same statistic, same
# population, paper_figures' global-RNG pair bootstrap), calibrated_attacks draws otherwise
d3 = lambda s: s[1:] if s.startswith("0.") else s
def cell_auc(k, row):
    src = {("orc", "ratio"): ("aucnatlo", "aucnathi"), ("orc", "z"): ("auczorcnatlo", "auczorcnathi"),
           ("sib", "z"): ("auczsibnatlo", "auczsibnathi"), ("pub", "z"): ("auczpubnatlo", "auczpubnathi")}.get((k, row))
    lo, hi = (pm[src[0]], pm[src[1]]) if src else (f"{V[f'table.{k}.{row}.auc_lo']:.3f}", f"{V[f'table.{k}.{row}.auc_hi']:.3f}")
    return f"{V[f'table.{k}.{row}.auc']:.3f}& {{\\scriptsize[{d3(lo)},{d3(hi)}]}}", bool(src)
def cell_tpr(k, row):
    src = {("orc", "ratio"): ("tpronenatlo", "tpronenathi"), ("orc", "z"): ("tpronezorcnatlo", "tpronezorcnathi"),
           ("sib", "z"): ("tpronezsibnatlo", "tpronezsibnathi"), ("pub", "z"): ("tpronezpubnatlo", "tpronezpubnathi")}.get((k, row))
    lo, hi = (pm[src[0]], pm[src[1]]) if src else (f"{V[f'table.{k}.{row}.tpr_lo']:.1f}", f"{V[f'table.{k}.{row}.tpr_hi']:.1f}")
    return f"{V[f'table.{k}.{row}.tpr']:.1f}& {{\\scriptsize[{lo},{hi}]}}", bool(src)
base = (HERE / "calibrated_allNAT1.tex").read_text().splitlines()
out = []; cur = None
for ln in base:
    if "emph{Exact" in ln: cur = "orc"
    elif "emph{Retrained" in ln: cur = "sib"
    elif "emph{Published" in ln: cur = "pub"
    if ln.startswith("\\quad Ratio") or ln.startswith("\\quad Standardized"):
        row = "ratio" if "Ratio &" in ln and "Standardized" not in ln else "z"
        lab = "Ratio" if row == "ratio" else "Standardized ratio"
        a, _ = cell_auc(cur, row); t, _ = cell_tpr(cur, row)
        ln = f"\\quad {lab} & {a} & {t} & {V[f'table.{cur}.{row}.strat']:.1f} \\\\"
    out.append(ln)
out[1] = ("% 2026-10-28: recomputed on all NAT1 pairs; CIs of the exact-counterfactual rows and of every standardized-ratio row "
          "are the prose's whole-NAT1 macros (\\aucnatlo.., \\tpronenatlo.., \\auc/tprone z{orc,sib,pub}natlo..); "
          "retrained/published Ratio CIs from calibrated_attacks' shared draws")
(HERE / "calibrated_allNAT1_textCIs.tex").write_text("\n".join(out) + "\n")
print((HERE / "macros_allNAT1.tex").read_text()[:2500]); print("\n".join(out))
