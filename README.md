# LitDialogue

Official code for the paper: **LitDialogue: A Dual-Module Framework for Character-Consistent Dialogue Generation in Interactive Visual Novels**.

This repo contains:
- **Dialogue Generation (GPT-2)**: multi-turn dialogue generation conditioned on character/speaker
- **Trigger Classification (BERT)**: narrative trigger detection / binary classification
- **CLI Demo**: interactive chatting with a chosen NPC

## Data & Copyright Notice
This repository **does NOT redistribute** copyrighted book text or illustrations.
You should prepare your own legally obtained source data and build the processed files locally.

## Environment
- Python 3.9+ recommended

Install dependencies:
```bash
pip install -r requirements.txt

Usage
Train: Dialogue Generation (GPT-2)

python scripts/train_dialogue.py \
  --data_dir /path/to/TLP_AVG_Dataset_v1/processed/splits \
  --model_name gpt2 \
  --output_dir outputs/dialogue_gpt2 \
  --lang en \
  --max_len 256 \
  --batch_size 2 \
  --num_train_epochs 3

Train: Trigger Classification (BERT)

python scripts/train_trigger.py \
  --data_dir /path/to/TLP_AVG_Dataset_v1/processed/splits \
  --model_name bert-base-uncased \
  --output_dir outputs/trigger_bert \
  --lang en \
  --max_len 256 \
  --batch_size 8 \
  --num_train_epochs 3

Run demo (CLI)

python scripts/play_cli.py --model_dir outputs/dialogue_gpt2 --npc_name "Fox"

