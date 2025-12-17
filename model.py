# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
)
from .prompts import SPECIAL_TOKENS

# ========= Class 1 =========
@dataclass
class TokenizerWithSpecials:
    tokenizer: PreTrainedTokenizerBase
    @classmethod
    def load_local(cls, path: str):
        # 尝试标准方式；失败则用本地 vocab/merges/tokenizer.json 兜底
        try:
            tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        except Exception:
            vocab = os.path.join(path, "vocab.json")
            merges = os.path.join(path, "merges.txt")
            tokfile = os.path.join(path, "tokenizer.json")
            if os.path.exists(tokfile):
                tok = PreTrainedTokenizerFast(tokenizer_file=tokfile)
            elif os.path.exists(vocab) and os.path.exists(merges):
                tok = PreTrainedTokenizerFast(tokenizer_file=None, vocab_file=vocab, merges_file=merges)
            else:
                raise FileNotFoundError(f"Missing tokenizer files in {path}")
            stm = os.path.join(path, "special_tokens_map.json")
            if os.path.exists(stm):
                with open(stm, "r", encoding="utf-8") as f:
                    m = json.load(f)
                tok.add_special_tokens(m)
            else:
                tok.add_special_tokens({
                    "additional_special_tokens": SPECIAL_TOKENS,
                    "pad_token":"<|pad|>","unk_token":"<|unk|>","bos_token":"<|bos|>","eos_token":"<|eos|>"
                })
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        tok.add_tokens([t for t in SPECIAL_TOKENS if t not in tok.get_vocab()], special_tokens=False)
        return cls(tok)

# ========= Class 2 =========
@dataclass
class AuxHeadConfig:
    num_style_classes: int = 4
    lambda_style: float = 0.0
    lambda_consistency: float = 0.0
    dropout: float = 0.1
    pooler_type: str = "last"  # "last" | "mean"

# ========= Class 3 =========
class SequencePooler(nn.Module):
    def __init__(self, mode: str = "last"):
        super().__init__()
        assert mode in ("last", "mean")
        self.mode = mode
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if self.mode == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.float().unsqueeze(-1)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        else:
            if attention_mask is None:
                return hidden_states[:, -1, :]
            idx = attention_mask.long().sum(dim=1) - 1
            idx = idx.clamp_min(0)
            return hidden_states[torch.arange(hidden_states.size(0)), idx, :]

# ========= Class 4 =========
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        return x + self.drop(h)

# ========= Class 5 =========
class StyleClassifierHead(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(dim, num_classes)
        self.ce = nn.CrossEntropyLoss()
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None):
        z = self.proj(x)
        logits = self.classifier(z)
        loss = None
        if target is not None:
            mask = (target >= 0)
            if mask.any():
                loss = self.ce(logits[mask], target[mask].long())
        return logits, loss

# ========= Class 6 =========
class ConsistencyScoreHead(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.reg = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.mse = nn.MSELoss()
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None):
        y = self.reg(x).squeeze(-1)
        loss = None
        if target is not None:
            mask = torch.isfinite(target)
            if mask.any():
                loss = self.mse(y[mask], target[mask].float())
        return y, loss

# ========= Class 7 =========
class CausalLMWithAuxHeads(nn.Module):
    """
    封装原始 Causal LM，主任务=语言建模；可选两个辅助头（风格分类/一致性回归）。
    """
    def __init__(self, base_model: nn.Module, hidden_size: int, aux_cfg: AuxHeadConfig):
        super().__init__()
        self.base_model = base_model
        self.aux_cfg = aux_cfg
        self.pooler = SequencePooler(aux_cfg.pooler_type)
        self.deep_enhance = nn.Sequential(
            ResidualMLPBlock(hidden_size, hidden_size * 4, aux_cfg.dropout),
            ResidualMLPBlock(hidden_size, hidden_size * 4, aux_cfg.dropout),
        )
        self.style_head = StyleClassifierHead(hidden_size, aux_cfg.num_style_classes, aux_cfg.dropout) \
            if aux_cfg.lambda_style > 0 else None
        self.consist_head = ConsistencyScoreHead(hidden_size, aux_cfg.dropout) \
            if aux_cfg.lambda_consistency > 0 else None

    @property
    def config(self):
        return self.base_model.config

    def forward(self, input_ids, attention_mask=None, labels=None,
                style_target=None, consistency_target=None, **kwargs) -> Dict[str, Any]:
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                              labels=labels, output_hidden_states=True, **kwargs)
        loss = out.loss
        last_hidden = out.hidden_states[-1]
        pooled = self.pooler(last_hidden, attention_mask)
        pooled = self.deep_enhance(pooled)

        aux_losses = {}
        if self.style_head is not None and style_target is not None:
            style_logits, style_loss = self.style_head(pooled, style_target)
            if style_loss is not None:
                loss = loss + self.aux_cfg.lambda_style * style_loss
                aux_losses["style_loss"] = style_loss.detach()
        else:
            style_logits = None

        if self.consist_head is not None and consistency_target is not None:
            consist_pred, consist_loss = self.consist_head(pooled, consistency_target)
            if consist_loss is not None:
                loss = loss + self.aux_cfg.lambda_consistency * consist_loss
                aux_losses["consistency_loss"] = consist_loss.detach()
        else:
            consist_pred = None

        return {
            "loss": loss,
            "logits": out.logits,
            "hidden_states": out.hidden_states,
            "aux": {
                "pooled": pooled,
                "style_logits": style_logits,
                "consistency_pred": consist_pred,
                **aux_losses
            }
        }

    # ---- 转发给底层 HF 模型的方法（兼容 Trainer/Generate/Grad-CKPT） ----
    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # 关闭 cache 以配合 checkpointing
        try:
            if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "use_cache"):
                self.base_model.config.use_cache = False
        except Exception:
            pass
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            try:
                return self.base_model.gradient_checkpointing_enable(*args, **kwargs)
            except TypeError:
                # 兼容旧签名
                return self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self, *args, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            try:
                return self.base_model.gradient_checkpointing_disable(*args, **kwargs)
            except TypeError:
                return self.base_model.gradient_checkpointing_disable()

# ========== 工具函数 ==========
def _infer_hidden_size_from_config(cfg) -> int:
    for key in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    return 768

def _load_base_model_and_tokenizer(model_path: str) -> Tuple[nn.Module, PreTrainedTokenizerBase]:
    tok = TokenizerWithSpecials.load_local(model_path).tokenizer
    try:
        base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        base.resize_token_embeddings(len(tok))
    except Exception:
        cfg = GPT2Config.from_pretrained(model_path, local_files_only=True)
        base = GPT2LMHeadModel(cfg); base.resize_token_embeddings(len(tok))
    return base, tok

def load_model_and_tokenizer(model_path: str):
    base, tok = _load_base_model_and_tokenizer(model_path)
    if os.getenv("LITGAME_USE_AUX", "0") != "1":
        return base, tok
    lambda_style = float(os.getenv("LITGAME_AUX_STYLE", "0.0"))
    lambda_cons = float(os.getenv("LITGAME_AUX_CONSIST", "0.0"))
    pooler = os.getenv("LITGAME_POOLER", "last")
    num_cls = int(os.getenv("LITGAME_STYLE_CLASSES", "4"))
    aux_cfg = AuxHeadConfig(
        num_style_classes=num_cls,
        lambda_style=lambda_style,
        lambda_consistency=lambda_cons,
        dropout=0.1,
        pooler_type=pooler
    )
    hidden = _infer_hidden_size_from_config(base.config)
    model = CausalLMWithAuxHeads(base, hidden, aux_cfg)
    return model, tok
