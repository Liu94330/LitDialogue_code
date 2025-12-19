import json, os, hashlib
def read_jsonl(path):
    out=[]; 
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
    return out
def get_any(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default
def autodiscover_splits(data_dir: str):
    cands=[("train","pairs_train.jsonl"),("val","pairs_val.jsonl"),("test","pairs_test.jsonl")]
    resolved={}
    for sp,fn in cands:
        p=os.path.join(data_dir,fn)
        if os.path.exists(p): resolved[sp]=p
    if not resolved:
        raise FileNotFoundError(f"No expected JSONL splits found in {data_dir}. Expected names: pairs_train/val/test.jsonl")
    return resolved
def stable_str_to_id(s: str, num_classes: int=5) -> int:
    if not s: return -1
    h=int(hashlib.md5(s.encode("utf-8")).hexdigest(),16)
    return h % max(1,num_classes)
