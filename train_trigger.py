import os
import argparse
import random
import numpy as np

from datasets import load_dataset, DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def build_text(example):
    """
    把 context 和 response 拼成一个文本字段 text，并且把 trigger_label 转成 labels（int）
    """
    ctx = example.get("context", [])
    ctx_str = ""
    if ctx:
        if isinstance(ctx, list):
            ctx_str = "\n".join(ctx)
        else:
            ctx_str = str(ctx)

    resp = example.get("response", "")
    resp = "" if resp is None else str(resp)

    if ctx_str:
        text = ctx_str + "\n" + resp
    else:
        text = resp

    label = int(example.get("trigger_label", 0))
    return {"text": text, "labels": label}


def tokenize_function(examples, tokenizer, max_len):
    """
    对 text 做分词，并保留 labels
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    tokenized["labels"] = examples["labels"]
    return tokenized


def compute_metrics(eval_pred):
    """
    评价指标：accuracy, precision, recall, f1（binary）
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to TLP_AVG_Dataset_v1/processed/splits",
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./outputs/trigger_bert")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    # 固定随机种子
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    # 构造文件路径
    train_file = os.path.join(args.data_dir, "pairs_train.jsonl")
    val_file = os.path.join(args.data_dir, "pairs_val.jsonl")
    test_file = os.path.join(args.data_dir, "pairs_test.jsonl")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Train file not found: {train_file}")

    raw_datasets = DatasetDict()

    # 1）加载 train
    print(f"Loading train split from: {train_file}")
    train_ds = load_dataset(
        "json",
        data_files=train_file,
        split="train",
    )
    raw_datasets["train"] = train_ds

    # 2）尝试加载 validation
    if os.path.exists(val_file) and os.path.getsize(val_file) > 0:
        try:
            print(f"Loading validation split from: {val_file}")
            val_ds = load_dataset(
                "json",
                data_files=val_file,
                split="train",
            )
            if len(val_ds) > 0:
                raw_datasets["validation"] = val_ds
            else:
                print("Validation file loaded but has 0 examples, will create from train.")
        except Exception as e:
            print(f"Warning: could not load validation from {val_file}: {e}")
    else:
        print("No non-empty validation file found, will create validation from train.")

    # 3）尝试加载 test
    if os.path.exists(test_file) and os.path.getsize(test_file) > 0:
        try:
            print(f"Loading test split from: {test_file}")
            test_ds = load_dataset(
                "json",
                data_files=test_file,
                split="train",
            )
            if len(test_ds) > 0:
                raw_datasets["test"] = test_ds
            else:
                print("Test file loaded but has 0 examples, will skip test.")
        except Exception as e:
            print(f"Warning: could not load test from {test_file}: {e}")
    else:
        print("No non-empty test file found, skipping test split.")

    # 4）按语言过滤
    def lang_filter(example):
        return example.get("language", "en") == args.lang

    raw_datasets = raw_datasets.filter(lang_filter)

    # 5）尽量把 trigger_label 转成 ClassLabel，并检查每个类别的数量
    can_stratify = False
    if "trigger_label" in raw_datasets["train"].column_names:
        try:
            if not isinstance(raw_datasets["train"].features["trigger_label"], ClassLabel):
                print("Encoding 'trigger_label' column as ClassLabel for stratified split.")
                raw_datasets["train"] = raw_datasets["train"].class_encode_column("trigger_label")

            if isinstance(raw_datasets["train"].features["trigger_label"], ClassLabel):
                labels = raw_datasets["train"]["trigger_label"]
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts = dict(zip(unique_labels.tolist(), counts.tolist()))
                print("Trigger_label class distribution in train:", label_counts)
                min_count = counts.min()
                if min_count >= 2:
                    can_stratify = True
                else:
                    print(f"Cannot stratify: minimum class count is {min_count} (< 2). Will use random split.")
        except Exception as e:
            print(f"Warning: could not analyze/encode 'trigger_label' for stratified split: {e}")
            can_stratify = False

    # 6）如果 validation 不存在或空，从 train 里切 10% 出来做验证集
    needs_new_val = (
        "validation" not in raw_datasets
        or len(raw_datasets["validation"]) == 0
    )
    if needs_new_val:
        print("No non-empty validation split after filtering, creating from train (10%).")
        if len(raw_datasets["train"]) < 2:
            raise ValueError(
                "Not enough training data to create a validation split."
            )
        if can_stratify:
            print("Using stratified split by 'trigger_label'.")
            split = raw_datasets["train"].train_test_split(
                test_size=0.1,
                seed=42,
                stratify_by_column="trigger_label",
            )
        else:
            print("Using random split (no stratification).")
            split = raw_datasets["train"].train_test_split(
                test_size=0.1,
                seed=42,
            )
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]

    print("Dataset sizes after filtering / splitting:")
    for split_name, ds in raw_datasets.items():
        print(f"  {split_name}: {len(ds)} examples")

    # 7）构造 text + labels
    with_text = raw_datasets.map(build_text)

    # 8）分词
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = with_text.map(
        lambda batch: tokenize_function(batch, tokenizer, args.max_len),
        batched=True,
        remove_columns=with_text["train"].column_names,
    )

    # 9）模型
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    # 10）训练参数（兼容老版本 transformers）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=50,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # 11）训练
    trainer.train()

    # 12）保存模型与 tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 13）验证集指标
    eval_results = trainer.evaluate()
    print("Validation metrics:", eval_results)

    # 14）测试集（如果存在且非空）
    if "test" in tokenized_datasets and len(tokenized_datasets["test"]) > 0:
        test_results = trainer.evaluate(
            eval_dataset=tokenized_datasets["test"]
        )
        print("Test metrics:", test_results)
    else:
        print("No non-empty test split found; skipping test evaluation.")


if __name__ == "__main__":
    main()
