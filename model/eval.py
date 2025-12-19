# -*- coding: utf-8 -*-
import os, argparse, json, math
from tqdm import tqdm
import torch
from .utils import autodiscover_splits, read_jsonl
from .data import example_to_text
from .model import load_model_and_tokenizer

def bleu1(pred, ref):
    p = pred.strip().split(); r = ref.strip().split()
    if not p or not r: return 0.0
    match = sum(1 for w in p if w in set(r))
    prec = match / len(p)
    bp = math.exp(1 - len(r)/len(p)) if len(p) < len(r) else 1.0
    return prec * bp

def rouge_l(pred, ref):
    def lcs(a, b):
        na, nb = len(a), len(b)
        dp = [[0]*(nb+1) for _ in range(na+1)]
        for i in range(na):
            for j in range(nb):
                if a[i]==b[j]: dp[i+1][j+1]=dp[i][j]+1
                else: dp[i+1][j+1]=max(dp[i][j+1], dp[i+1][j])
        return dp[na][nb]
    a, b = pred.strip().split(), ref.strip().split()
    if not a or not b: return 0.0
    l = lcs(a,b); recall = l/len(b); prec = l/len(a)
    return 0.0 if recall+prec==0 else (2*recall*prec)/(recall+prec)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",type=str,required=True)
    ap.add_argument("--ckpt",type=str,required=True)
    ap.add_argument("--split",type=str,default="val",choices=["train","val","test"])
    ap.add_argument("--max_new_tokens",type=int,default=96)
    ap.add_argument("--num_eval",type=int,default=200)
    args=ap.parse_args()

    splits=autodiscover_splits(args.data_dir)
    rows=read_jsonl(splits[args.split])[:args.num_eval]
    model, tokenizer = load_model_and_tokenizer(args.ckpt); model.eval()

    preds, refs = [], []
    for r in tqdm(rows,desc="Evaluating"):
        ex=example_to_text(r)
        prompt, target = ex["prompt"], ex["target"]
        inputs=tokenizer(prompt,return_tensors="pt")
        inputs={k:v.to(model.device) for k,v in inputs.items()}
        with torch.no_grad():
            out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=True,top_p=0.9,temperature=0.8,
                               pad_token_id=tokenizer.pad_token_id)
        gen=tokenizer.decode(out[0],skip_special_tokens=True)
        if "<|assistant|>" in gen: gen=gen.split("<|assistant|>")[-1].strip()
        preds.append(gen); refs.append(target)

    bleu = sum(bleu1(p,r) for p,r in zip(preds,refs))/max(1,len(preds))
    rouge = sum(rouge_l(p,r) for p,r in zip(preds,refs))/max(1,len(preds))
    with open(os.path.join(args.ckpt,f"metrics_{args.split}.json"),"w",encoding="utf-8") as f:
        json.dump({"bleu1":bleu,"rougeL":rouge},f,ensure_ascii=False,indent=2)
    print("Saved metrics to", os.path.join(args.ckpt,f"metrics_{args.split}.json"))

if __name__=="__main__":
    main()
