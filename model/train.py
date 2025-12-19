# -*- coding: utf-8 -*-
import os, math, argparse, inspect
from datasets import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
import torch

from .utils import read_jsonl, autodiscover_splits
from .data import example_to_text, tokenize_and_mask, DataCollatorForCausalLMSelective
from .model import load_model_and_tokenizer

class EMACallback(TrainerCallback):
    def __init__(self, decay=0.999):
        self.decay=decay; self.shadow={}; self.backup={}; self.enabled=False
    def on_train_begin(self,args,state,control,**kw):
        m=kw["model"]; self.shadow={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}; self.enabled=True
    def on_step_end(self,args,state,control,**kw):
        if not self.enabled: return
        m=kw["model"]
        with torch.no_grad():
            for n,p in m.named_parameters():
                if not p.requires_grad: continue
                self.shadow[n].mul_(self.decay).add_(p.detach(),alpha=1.0-self.decay)
    def _apply(self,m):
        self.backup={}
        for n,p in m.named_parameters():
            if not p.requires_grad: continue
            self.backup[n]=p.detach().clone(); p.data.copy_(self.shadow[n].data)
    def _restore(self,m):
        for n,p in m.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n].data)
        self.backup={}
    def on_evaluate(self,args,state,control,**kw):
        if self.enabled: self._apply(kw["model"])
    def on_evaluate_end(self,args,state,control,**kw):
        if self.enabled: self._restore(kw["model"])
    def on_save(self,args,state,control,**kw):
        if self.enabled: self._apply(kw["model"])
    def on_save_end(self,args,state,control,**kw):
        if self.enabled: self._restore(kw["model"])

def build_dataset(path: str, style_classes: int):
    rows=read_jsonl(path)
    examples=[example_to_text(r, style_classes=style_classes) for r in rows]
    return Dataset.from_list(examples)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",type=str,required=True)
    ap.add_argument("--out_dir",type=str,required=True)
    ap.add_argument("--model_name",type=str,required=True)
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--batch_size",type=int,default=4)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--gradient_accumulation",type=int,default=1)
    ap.add_argument("--fp16",action="store_true")
    ap.add_argument("--bf16",action="store_true")
    ap.add_argument("--max_length",type=int,default=512)
    ap.add_argument("--max_new_tokens",type=int,default=96)
    # 复杂训练参数
    ap.add_argument("--weight_decay",type=float,default=0.1)
    ap.add_argument("--warmup_ratio",type=float,default=0.03)
    ap.add_argument("--scheduler",type=str,default="cosine",choices=["linear","cosine","poly","constant"])
    ap.add_argument("--clip_grad",type=float,default=1.0)
    ap.add_argument("--grad_ckpt",action="store_true")
    ap.add_argument("--ema",action="store_true")
    ap.add_argument("--ema_decay",type=float,default=0.999)
    # 可选自定义split
    ap.add_argument("--train_file",type=str,default=None)
    ap.add_argument("--val_file",type=str,default=None)
    ap.add_argument("--test_file",type=str,default=None)
    # 多任务类别数
    ap.add_argument("--style_classes",type=int,default=int(os.getenv("LITGAME_STYLE_CLASSES","5")))
    args=ap.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)

    # 选择数据 split
    if args.train_file and args.val_file and args.test_file:
        splits={"train":args.train_file,"val":args.val_file,"test":args.test_file}
    else:
        splits=autodiscover_splits(args.data_dir)
    dtrain=build_dataset(splits.get("train",list(splits.values())[0]), style_classes=args.style_classes)
    dval=build_dataset(splits.get("val",list(splits.values())[-1]), style_classes=args.style_classes)

    # 模型与分词器
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # 梯度检查点：禁用cache + 调用底层可用方法
    if args.grad_ckpt:
        for m in (model, getattr(model, "base_model", None)):
            if m is None: continue
            try:
                if hasattr(m, "config") and hasattr(m.config, "use_cache"):
                    m.config.use_cache = False
            except Exception: pass
            try:
                if hasattr(m, "gradient_checkpointing_enable"):
                    m.gradient_checkpointing_enable()
            except Exception: pass

    # 构建特征
    def proc(b):
        out=tokenize_and_mask(tokenizer,b["prompt"],b["target"],max_length=args.max_length)
        if "style_target" in b: out["style_target"]=b["style_target"]
        if "consistency_target" in b: out["consistency_target"]=b["consistency_target"]
        return out
    dtrain=dtrain.map(proc,batched=False,desc="Tokenizing train")
    dval=dval.map(proc,batched=False,desc="Tokenizing val")

    collator=DataCollatorForCausalLMSelective(tokenizer)

    # --------- 兼容不同版本的 TrainingArguments ----------
    sig = inspect.signature(TrainingArguments.__init__)
    valid = set(sig.parameters.keys())

    # 估算 warmup_steps（老版本没有 warmup_ratio）
    steps_per_epoch = max(1, math.ceil(len(dtrain) / max(1, args.batch_size * max(1, args.gradient_accumulation))))
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)

    # 先准备一个“新版本风格”的参数表
    ta_kwargs = dict(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        logging_steps=100,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=args.clip_grad,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        gradient_checkpointing=args.grad_ckpt,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
    )

    # 只传它认识的
    kwargs = {k: v for k, v in ta_kwargs.items() if k in valid}

    # 若不支持 evaluation_strategy，则：回退老接口 + 禁用 load_best_model_at_end 以避免策略不匹配
    if "evaluation_strategy" not in valid:
        if "evaluate_during_training" in valid:
            kwargs["evaluate_during_training"] = True
        # 有些老版本没有 warmup_ratio，用 warmup_steps 代替
        if "warmup_ratio" not in valid and "warmup_steps" in valid:
            kwargs["warmup_steps"] = warmup_steps
        # 避免触发“eval/save策略不匹配”的校验
        if "load_best_model_at_end" in kwargs:
            kwargs["load_best_model_at_end"] = False
        # 部分非常老的版本没有 lr_scheduler_type
        if "lr_scheduler_type" not in valid and "lr_schedule" in valid:
            m = {"cosine": "cosine", "linear": "linear", "constant": "constant"}
            kwargs["lr_schedule"] = m.get(args.scheduler, "linear")

    training_args = TrainingArguments(**kwargs)
    # -------------------------------------------------------

    callbacks=[EMACallback(decay=args.ema_decay)] if args.ema else None
    trainer=Trainer(model=model, args=training_args,
                    train_dataset=dtrain, eval_dataset=dval,
                    data_collator=collator, tokenizer=tokenizer,
                    callbacks=callbacks)

    trainer.train()
    ev=trainer.evaluate()
    ppl=math.exp(ev["eval_loss"]) if "eval_loss" in ev else float("nan")
    with open(os.path.join(args.out_dir,"eval_summary.txt"),"w",encoding="utf-8") as f:
        f.write(f"eval_loss={ev.get('eval_loss'):.4f}\nperplexity={ppl:.4f}\n")
    trainer.save_model(args.out_dir); tokenizer.save_pretrained(args.out_dir)

if __name__=="__main__":
    main()
