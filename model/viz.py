# -*- coding: utf-8 -*-
import os, json, argparse, math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def _smooth(y, k=15):
    if k <= 1 or len(y) < k: return y
    out=[]; s=0.0
    from collections import deque
    q=deque()
    for v in y:
        q.append(v); s+=v
        if len(q)>k: s-=q.popleft()
        out.append(s/len(q))
    return out
def _plot_xy(steps, values, outpath, title, ylabel, smooth_k=15):
    plt.figure(figsize=(8,4.5))
    plt.plot(steps, values, label="raw", alpha=0.5)
    plt.plot(steps, _smooth(values, k=smooth_k), label=f"smooth(k={smooth_k})")
    plt.title(title); plt.xlabel("step"); plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.3); plt.legend()
    ax = plt.gca(); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(outpath, dpi=180); plt.close()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run_dir",type=str,required=True); args=ap.parse_args()
    st=os.path.join(args.run_dir,"trainer_state.json")
    if not os.path.exists(st): raise FileNotFoundError(st)
    state=json.load(open(st,"r",encoding="utf-8")); logs=state.get("log_history",[])
    steps=[]; train_loss=[]; eval_loss=[]; lrs=[]; others={}; num_keys=set()
    for r in logs:
        if "step" not in r: continue
        steps.append(r["step"]); train_loss.append(r.get("loss", float("nan")))
        eval_loss.append(r.get("eval_loss", float("nan"))); lrs.append(r.get("learning_rate", float("nan")))
        for k,v in r.items():
            if k in ("step","loss","eval_loss","learning_rate"): continue
            if isinstance(v,(int,float)): num_keys.add(k); others.setdefault(k,[]).append(v)
    if any(x==x for x in train_loss): _plot_xy(steps, train_loss, os.path.join(args.run_dir,"train_loss.png"), "Train Loss vs Step", "loss")
    if any(x==x for x in eval_loss):
        _plot_xy(steps, eval_loss, os.path.join(args.run_dir,"eval_loss.png"), "Eval Loss vs Step", "eval_loss")
        ppl=[math.exp(v) if v==v else float("nan") for v in eval_loss]
        with open(os.path.join(args.run_dir,"eval_ppl_last.txt"),"w",encoding="utf-8") as f:
            if any(p==p for p in ppl): f.write(f"last_eval_ppl={[p for p in ppl if p==p][-1]:.4f}\n")
            else: f.write("last_eval_ppl=NaN\n")
    if any(x==x for x in lrs): _plot_xy(steps, lrs, os.path.join(args.run_dir,"lr.png"), "Learning Rate vs Step", "lr", smooth_k=1)
    if num_keys:
        plt.figure(figsize=(9,5))
        for k in sorted(num_keys): plt.plot(steps, others[k], label=k, alpha=0.85)
        plt.title("All Numeric Metrics"); plt.xlabel("step"); plt.ylabel("value")
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend(loc="best", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(args.run_dir,"all_metrics.png"), dpi=180); plt.close()
    print("Saved figures to:", args.run_dir)
if __name__=="__main__": main()
