import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def generate_reply(model, tokenizer, history, npc_name="Little Prince",
                   max_new_tokens=64, temperature=0.8, device="cpu"):
    # 只保留最近 8 轮，避免上下文太长
    history = history[-8:]
    prompt = "\n".join(history + [f"{npc_name}:"])
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 提取最后一次 npc_name: 之后的内容
    last_marker = full_text.rfind(f"{npc_name}:")
    if last_marker != -1:
        reply_text = full_text[last_marker + len(f"{npc_name}:"):].strip()
    else:
        # fallback：直接截取在 prompt 后面的新增部分
        reply_text = full_text[len(prompt):].strip()
    # 只要第一行，防止跑飞
    reply_text = reply_text.split("\n")[0].strip()
    return reply_text, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to fine-tuned dialogue model (from train_dialogue.py)")
    parser.add_argument("--npc_name", type=str, default="Little Prince",
                        help="Character to talk to: 'Little Prince', 'Fox', 'Rose', etc.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== The Little Prince AVG Demo ===")
    print(f"You are now talking with: {args.npc_name}")
    print("Type 'quit' to exit.\n")

    history = [
        "NARRATOR: You are inside an interactive adaptation of 'The Little Prince'.",
        f"{args.npc_name}: Hello. Who are you?",
    ]
    print(f"{args.npc_name}: Hello. Who are you?")

    while True:
        user = input("You: ").strip()
        if user.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if not user:
            continue
        history.append(f"Player: {user}")
        reply, history = generate_reply(
            model, tokenizer, history,
            npc_name=args.npc_name,
            device=device,
        )
        print(f"{args.npc_name}: {reply}")
        history.append(f"{args.npc_name}: {reply}")


if __name__ == "__main__":
    main()
