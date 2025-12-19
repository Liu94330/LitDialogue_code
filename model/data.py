from typing import Dict, Any, List
from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizerBase
from .utils import get_any, stable_str_to_id
from .prompts import build_prompt

SRC_KEYS=["context","source","prompt","input","history"]
TGT_KEYS=["response","target","label","output","reply"]
ROLE_KEYS=["role","character"]
SCENE_KEYS=["scene","location"]
STYLE_ID_KEYS=["style_id","style_label_id"]
STYLE_STR_KEYS=["style","emotion","tone","style_name"]
CONS_KEYS=["consistency_score","cons_score","role_consistency"]

def example_to_text(ex: Dict[str, Any], style_classes: int=5) -> Dict[str, Any]:
    src=get_any(ex,SRC_KEYS,""); tgt=get_any(ex,TGT_KEYS,"")
    role=get_any(ex,ROLE_KEYS,"未知角色"); scene=get_any(ex,SCENE_KEYS,"默认场景")
    if isinstance(src,list):
        lines=[]
        for turn in src:
            if isinstance(turn,dict):
                who=turn.get("role","未指明"); txt=turn.get("text") or turn.get("content") or ""
                lines.append(f"{who}：{txt}")
            else: lines.append(str(turn))
        src="\n".join(lines)
    style_target=-1
    st_id=get_any(ex,STYLE_ID_KEYS,None)
    if isinstance(st_id,int):
        style_target=st_id
    else:
        st_str=get_any(ex,STYLE_STR_KEYS,None)
        if isinstance(st_str,str):
            style_target=stable_str_to_id(st_str, style_classes)
    cons=float("nan")
    c=get_any(ex,CONS_KEYS,None)
    if c is not None:
        try: cons=float(c)
        except: pass
    prompt_text=build_prompt(role,scene,context=src,player_text="")
    target_text=str(tgt) if tgt is not None else ""
    return {"prompt":prompt_text,"target":target_text,"role":role,"scene":scene,
            "style_target": style_target, "consistency_target": cons}

@dataclass
class DataCollatorForCausalLMSelective:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool=False
    def __call__(self, features: List[Dict[str, Any]]):
        input_ids=[torch.tensor(f["input_ids"],dtype=torch.long) for f in features]
        labels=[torch.tensor(f["labels"],dtype=torch.long) for f in features]
        attn=[torch.ones_like(x) for x in input_ids]
        input_ids=torch.nn.utils.rnn.pad_sequence(input_ids,batch_first=True,padding_value=self.tokenizer.pad_token_id)
        labels=torch.nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-100)
        attn=torch.nn.utils.rnn.pad_sequence(attn,batch_first=True,padding_value=0)
        batch={"input_ids":input_ids,"labels":labels,"attention_mask":attn}
        st=torch.tensor([f.get("style_target",-1) for f in features],dtype=torch.long)
        if (st>=0).any(): batch["style_target"]=st
        cv=torch.tensor([f.get("consistency_target", float("nan")) for f in features],dtype=torch.float)
        if torch.isfinite(cv).any(): batch["consistency_target"]=cv
        return batch

def tokenize_and_mask(tokenizer: PreTrainedTokenizerBase, prompt: str, target: str, max_length: int):
    full_text=prompt+target
    enc=tokenizer(full_text,truncation=True,max_length=max_length)
    input_ids=enc["input_ids"]
    tgt_ids=tokenizer(target,add_special_tokens=False)["input_ids"]
    start_idx=None
    for i in range(len(input_ids)-len(tgt_ids),-1,-1):
        if input_ids[i:i+len(tgt_ids)]==tgt_ids:
            start_idx=i; break
    labels=[-100]*len(input_ids)
    if start_idx is not None and len(tgt_ids)>0:
        labels[start_idx:start_idx+len(tgt_ids)]=tgt_ids
    return {"input_ids":input_ids,"labels":labels}
