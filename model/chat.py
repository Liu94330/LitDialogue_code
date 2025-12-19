# -*- coding: utf-8 -*-
import argparse, torch
from .model import load_model_and_tokenizer
from .prompts import build_prompt
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",type=str,required=True)
    ap.add_argument("--role",type=str,default="玫瑰")
    ap.add_argument("--scene",type=str,default="B612")
    ap.add_argument("--max_new_tokens",type=int,default=128)
    args=ap.parse_args()
    model, tokenizer = load_model_and_tokenizer(args.ckpt); model.eval()
    history=[]; print(f"[Chat] 角色={args.role} 场景={args.scene}，Ctrl+C 退出")
    while True:
        try: user=input("你：").strip()
        except KeyboardInterrupt: print("\nBye."); break
        if not user: continue
        ctx="\n".join(history[-8:])
        prompt=build_prompt(args.role,args.scene,context=ctx,player_text=user)
        inputs=tokenizer(prompt,return_tensors="pt")
        inputs={k:v.to(model.device) for k,v in inputs.items()}
        with torch.no_grad():
            out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=True,top_p=0.9,temperature=0.8,
                               pad_token_id=tokenizer.pad_token_id)
        txt=tokenizer.decode(out[0],skip_special_tokens=True)
        reply=txt.split("<|assistant|>")[-1].strip()
        print(f"{args.role}：{reply}")
        history.append(f"玩家：{user}"); history.append(f"{args.role}：{reply}")
if __name__=="__main__": main()
