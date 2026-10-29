"""Positive control: the re-check pipeline (decode scored tokens -> normalize ->
8-word shingles -> longest run) must recover known SFT hits from the 2026-09-29
audit (posttrain_overlap/posttrain_hits.csv.gz under the data root).

Needs the token files under bundle/tokens/, the OLMo 2 tokenizer in
assets/hf_models/OLMo-2-0425-1B and the SFT dataset snapshot in
assets/posttrain_datasets/ (download_posttrain_datasets.py)."""
import glob, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, str(Path(__file__).resolve().parent))
import olmo2_nonmember_audit as na, build_posttrain_corpus as bpc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths
from transformers import AutoTokenizer
import gzip, csv
B = paths.BUNDLE
tok = AutoTokenizer.from_pretrained(str(paths.ASSETS / "hf_models/OLMo-2-0425-1B"))
ev = {r["doc_id"]: r for r in csv.DictReader(gzip.open(B / "evaluation_manifest.csv.gz", "rt"))}
want = {"ai2-adapt-dev/flan_v2_converted_13431": ("wikiB:18250951", 181, 174),
        "ai2-adapt-dev/numinamath_tir_math_decontaminated_56250": ("2507.02833", 87, 80),
        "ai2-adapt-dev/flan_v2_converted_84762": ("wikiB:31874", 82, 159)}
files = sorted(glob.glob(str(paths.ASSETS / "posttrain_datasets/tulu-3-sft-olmo-2-mixture-0225/data/*.parquet")))
for f in files:
    t = pq.read_table(f)
    t = t.filter(pc.is_in(t["id"], value_set=__import__("pyarrow").array(list(want))))
    for row in t.to_pylist():
        doc, exp_run, exp_sh = want[row["id"]]
        rec = next(bpc.records_from_row("sft", row, 0))
        rh, rpos, rkept = na.shingles_filtered(na.normalize(rec["text"]), 8, na.MIN_SHINGLE_CHARS)
        ids = np.load(B / ev[doc]["tokens_relpath"]).astype(np.int64).tolist()
        ch, cpos, ckept = na.shingles_filtered(na.normalize(tok.decode(ids, skip_special_tokens=True)), 8, na.MIN_SHINGLE_CHARS)
        shared = np.intersect1d(rh, ch)
        _, run_r, _ = na.longest_run(rpos[np.isin(rh, shared)], rkept, 8)
        print(f"{doc:16s} vs {row['id']}: shingles {len(shared)} (audit {exp_sh}), run {run_r} words (audit {exp_run})")
