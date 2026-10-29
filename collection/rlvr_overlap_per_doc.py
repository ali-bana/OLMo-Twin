"""One row per evaluation document that shares at least one 8-word shingle with
an RLVR record (input of the RLVR sensitivity rows).

Reads rlvr/overlap/rlvr_overlap_all_matches.csv.gz (written by
rlvr_overlap_scan.py) under the data root and writes
rlvr/overlap/rlvr_overlap_per_doc.csv.gz next to it; prints the documents with
at least 5 shared shingles or a shared run of at least 20 words.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

OUT = paths.RLVR / "overlap"

import pandas as pd
d=pd.read_csv(OUT / 'rlvr_overlap_all_matches.csv.gz',keep_default_na=False)
d['run']=d[['run_words_record','run_words_doc']].max(axis=1)
nat={'M','L','W-new','W-gap','S-se'}
d['NAT1']=d.cohort.isin(nat)&(d.r.astype(str)=='1')
g=d.groupby(['doc_id','cohort','arm','pair_id','NAT1']).agg(max_shingles=('shingles','max'),max_run=('run','max'),
   n_records=('record_id','nunique'),components=('component',lambda s:'|'.join(sorted(set(s))))).reset_index()
top=d.sort_values(['shingles','run'],ascending=False).drop_duplicates('doc_id').set_index('doc_id')
g['best_record']=g.doc_id.map(top.record_id); g['excerpt']=g.doc_id.map(top.excerpt)
g=g.sort_values(['max_shingles','max_run'],ascending=False)
g.to_csv(OUT / 'rlvr_overlap_per_doc.csv.gz',index=False)
pd.set_option('display.width',260); pd.set_option('display.max_colwidth',90)
s=g[(g.max_shingles>=5)|(g.max_run>=20)]
print(len(s)); print(s.drop(columns=['components']).to_string(index=False))
print(g.groupby(['cohort','arm']).size().to_string())
print('NAT1 docs', g.NAT1.sum(), 'NAT1 pairs touched', g[g.NAT1].pair_id.nunique())
