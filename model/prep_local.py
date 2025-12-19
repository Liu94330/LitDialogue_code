import os, argparse
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

from .utils import autodiscover_splits, read_jsonl
from .data import SRC_KEYS, TGT_KEYS, ROLE_KEYS, SCENE_KEYS
from .prompts import SPECIAL_TOKENS

def collect_corpus(data_dir, train_file=None, val_file=None, test_file=None):
    if train_file and val_file and test_file:
        split_paths={"train":train_file,"val":val_file,"test":test_file}
    else:
        split_paths=autodiscover_splits(data_dir)
    corpus=[]
    for _,p in split_paths.items():
        rows=read_jsonl(p)
        for d in rows:
            for keys in (SRC_KEYS,TGT_KEYS):
                for k in keys:
                    v=d.get(k)
                    if v is None: continue
                    if isinstance(v,list): v=" ".join(str(x) for x in v)
                    corpus.append(str(v))
            for k in ROLE_KEYS+SCENE_KEYS:
                if d.get(k): corpus.append(str(d[k]))
    return corpus

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",type=str,required=True)
    ap.add_argument("--out_dir",type=str,required=True)
    ap.add_argument("--train_file",type=str,default=None)
    ap.add_argument("--val_file",type=str,default=None)
    ap.add_argument("--test_file",type=str,default=None)
    ap.add_argument("--vocab_size",type=int,default=32000)
    ap.add_argument("--min_freq",type=int,default=2)
    ap.add_argument("--n_layer",type=int,default=12)
    ap.add_argument("--n_head",type=int,default=12)
    ap.add_argument("--n_embd",type=int,default=768)
    ap.add_argument("--max_positions",type=int,default=1024)
    args=ap.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)
    corpus=collect_corpus(args.data_dir,args.train_file,args.val_file,args.test_file)
    corpus_path=os.path.join(args.out_dir,"corpus.txt")
    with open(corpus_path,"w",encoding="utf-8") as f:
        f.write("\n".join(corpus))

    # 训练 ByteLevel BPE
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[corpus_path],
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        special_tokens=SPECIAL_TOKENS + ["<|pad|>","<|unk|>","<|bos|>","<|eos|>"]
    )

    # 兼容：保存 vocab.json / merges.txt
    tok.save_model(args.out_dir)

    # 关键：额外保存 fast 分词器 JSON，避免触发 tiktoken 依赖
    tokenizer_json = os.path.join(args.out_dir, "tokenizer.json")
    tok.save(tokenizer_json)

    # 直接用 tokenizer.json 构造 GPT2TokenizerFast
    hf_tok = GPT2TokenizerFast(tokenizer_file=tokenizer_json)
    hf_tok.add_special_tokens({
        "additional_special_tokens": SPECIAL_TOKENS,
        "pad_token":"<|pad|>",
        "unk_token":"<|unk|>",
        "bos_token":"<|bos|>",
        "eos_token":"<|eos|>"
    })
    # 保存为 HuggingFace 目录结构（含 tokenizer_config.json / special_tokens_map.json）
    hf_tok.save_pretrained(args.out_dir)

    # 初始化随机 GPT-2 并保存
    cfg=GPT2Config(
        vocab_size=len(hf_tok),
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.max_positions,
        n_ctx=args.max_positions,
        bos_token_id=hf_tok.bos_token_id,
        eos_token_id=hf_tok.eos_token_id
    )
    model=GPT2LMHeadModel(cfg)
    model.resize_token_embeddings(len(hf_tok))
    model.save_pretrained(args.out_dir)
    print("Local GPT-2 ready at:", args.out_dir)

if __name__=="__main__":
    main()
