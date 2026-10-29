"""Check the learning rate logged by the mid-training run against its schedule.

Usage:
    python training/checks/check_lr_schedule.py RUN.wandb

RUN.wandb is one offline Weights & Biases log of the run. OLMo writes these
under <save_folder>/wandb/wandb/offline-run-<date>-<id>/run-<id>.wandb, i.e.
$OLMO_TWIN_DATA/run/checkpoints/wandb/wandb/... for our run (one directory per
launch that logged to W&B). Run the script once per file.

For every logged update u the learning rate must equal
7.4487e-5 * (1 - u / 23842): OLMo derives the horizon of the linear decay from
the duration of 5e10 tokens, so the rate reaches zero at update 23842 and stays
zero for the last 11 of the 23,852 updates (Appendix D). The script prints the
rate at a few updates and the number of logged updates that differ.

Optional check; needs the ``wandb`` package (it reads the log with wandb's
internal datastore reader) and the run's offline log.
"""
import sys, json
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
f=sys.argv[1]
ds=datastore.DataStore(); ds.open_for_scan(f)
want={"optim/learning_rate_group0","_step","train/ZLoss","train/CrossEntropyLoss"}
out={}
n=0
while True:
    try: d=ds.scan_data()
    except Exception as e: print("scan err",e); break
    if d is None: break
    r=pb.Record(); r.ParseFromString(d)
    if r.WhichOneof("record_type")=="history":
        row={}
        for it in r.history.item:
            k=it.key or "/".join(it.nested_key)
            if k in want: row[k]=json.loads(it.value_json)
        if "_step" in row and "optim/learning_rate_group0" in row:
            out[row["_step"]]=row
        n+=1
ks=sorted(out)
print("history rows",n,"with lr",len(ks),"first",ks[:1],"last",ks[-1:])
for k in [1,2,1000,ks[0],ks[len(ks)//2],23840,23841,23842,23843,23852]:
    if k in out:
        lr=out[k]["optim/learning_rate_group0"]; exp=7.4487e-05*(1-k/23842) if k<23842 else 0.0
        print(k,lr,"expected",exp, "ok" if abs(lr-exp)<=1e-12+1e-6*exp else "DIFF")
bad=[k for k in ks if abs(out[k]["optim/learning_rate_group0"]-(7.4487e-05*(1-k/23842) if k<23842 else 0.0))>1e-12+1e-6*7.4487e-05]
print("rows checked",len(ks),"mismatch",len(bad),bad[:5])
