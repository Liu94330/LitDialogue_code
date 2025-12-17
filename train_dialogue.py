import os
import argparse
import random
import numpy as np

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)


def build_text(example):
    """
    把一条样本转换成一个连续的多轮对话文本：
    context:
        ["SpeakerA: hi", "SpeakerB: hello", ...]
    response_speaker:
        "Narrator" / "Player" / ...
    response:
        "具体回复内容"
    => "SpeakerA: hi\nSpeakerB: hello\nNarrator: 具体回复内容"
    """
    ctx = example.get("context", [])
    ctx_str = ""
    if ctx:
        ctx_str = "\n".join(ctx)

    speaker = example.get("response_speaker", "Unknown")
    resp = example.get("response", "")

    if ctx_str:
        text = ctx_str + f"\n{speaker}: " + resp
    else:
        text = f"{speaker}: " + resp

    return {"text": text}


def tokenize_function(examples, tokenizer, max_len):
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    # Causal LM: labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to TLP_AVG_Dataset_v1/processed/splits",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="可以是 huggingface 名称 (e.g., gpt2) 或本地目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/dialogue_gpt2",
    )
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument(
        "--validation_split_percentage",
        type=int,
        default=10,
        help=(
            "当没有现成的 pairs_val.jsonl 时，从训练集按该比例划分验证集；"
            "如果存在 pairs_val.jsonl，则优先使用文件中的验证集。"
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "启用 huggingface 离线模式：不访问网络，只使用本地缓存或本地模型目录。"
        ),
    )

    args = parser.parse_args()

    # ====== 设置离线模式（如果需要） ======
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[INFO] Running in OFFLINE mode (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)")

    # 固定随机种子，方便论文写“可复现性”
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    train_path = os.path.join(args.data_dir, "pairs_train.jsonl")
    val_path = os.path.join(args.data_dir, "pairs_val.jsonl")
    test_path = os.path.join(args.data_dir, "pairs_test.jsonl")

    # 构建 data_files，先只保证 train 存在
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")

    data_files = {"train": train_path}

    # 如果有 test 文件就加上
    if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
        data_files["test"] = test_path

    # 如果有非空的 val 文件就加上；否则稍后用 train 划分
    has_val_file = os.path.exists(val_path) and os.path.getsize(val_path) > 0
    if has_val_file:
        data_files["validation"] = val_path

    # 读取原始数据集
    raw_datasets = load_dataset("json", data_files=data_files)

    # 如果没有 validation split，就从 train 里切一部分出来
    if "validation" not in raw_datasets:
        val_ratio = args.validation_split_percentage / 100.0
        split = raw_datasets["train"].train_test_split(
            test_size=val_ratio,
            seed=42,
        )
        train_dataset = split["train"]
        val_dataset = split["test"]

        # test：如果文件有就用文件里的；没有就先用和 val 一样的（方便算 PPL）
        if "test" in raw_datasets:
            test_dataset = raw_datasets["test"]
        else:
            test_dataset = val_dataset

        raw_datasets = DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset,
            }
        )

    # 只保留指定语言的样本
    def lang_filter(example):
        return example.get("language", "en") == args.lang

    raw_datasets = raw_datasets.filter(lang_filter)

    # 构造带 speaker 的文本
    with_text = raw_datasets.map(build_text)

    # ====== 加载 tokenizer & model（支持离线） ======
    print(f"[INFO] Loading tokenizer from {args.model_name} (offline={args.offline})")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=args.offline,
    )
    # GPT-2 默认无 pad_token，用 eos_token 补齐
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model from {args.model_name} (offline={args.offline})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.offline,
    )
    model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = with_text.map(
        lambda batch: tokenize_function(batch, tokenizer, args.max_len),
        batched=True,
        remove_columns=with_text["train"].column_names,
    )

    # ====== TrainingArguments：新旧 transformers 兼容处理 ======
    base_training_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=50,
    )

    try:
        # 新版 transformers（支持 evaluation_strategy/save_strategy）
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=False,
            report_to="none",
            **base_training_kwargs,
        )
    except TypeError:
        # 旧版 transformers：不认识 evaluation_strategy 等参数
        print(
            "[WARN] 当前 transformers 版本不支持 'evaluation_strategy' 或 'save_strategy'，"
            "将使用简化版 TrainingArguments（只在显式调用 trainer.evaluate() 时评估）。"
        )
        try:
            training_args = TrainingArguments(**base_training_kwargs)
        except TypeError:
            # 极老版本再兜底一层（只用最基本的参数）
            print(
                "[WARN] TrainingArguments 仍然报 TypeError，尝试使用最小参数集合 "
                "(output_dir, per_device_train_batch_size, num_train_epochs)。"
            )
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=args.num_train_epochs,
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # 训练
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 验证集 perplexity
    eval_results = trainer.evaluate()
    print("Eval loss:", eval_results.get("eval_loss"))
    if "eval_loss" in eval_results:
        try:
            ppl = float(np.exp(eval_results["eval_loss"]))
            print("Perplexity:", ppl)
        except OverflowError:
            print("Perplexity overflow, loss too large.")

    # 测试集 perplexity
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    print("Test loss:", test_results.get("eval_loss"))
    if "eval_loss" in test_results:
        try:
            ppl = float(np.exp(test_results["eval_loss"]))
            print("Test perplexity:", ppl)
        except OverflowError:
            print("Test perplexity overflow, loss too large.")


if __name__ == "__main__":
    main()
