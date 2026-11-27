"""FT-Transformer (Gorishniy et al., 2021, "Revisiting Deep Learning Models for
Tabular Data"), written directly in PyTorch so every piece is visible.

Architecture:
- Feature tokenizer: every feature becomes a d-dimensional token.
  numeric j  -> x_j * W_j + b_j          (one learned vector per feature)
  category j -> Embedding_j[level] + b_j
- A learned [CLS] token is prepended; L pre-norm Transformer blocks
  (multi-head self-attention + ReGLU feed-forward) mix the tokens.
- The prediction is read from the final [CLS] token: LayerNorm -> ReLU -> Linear.

Preprocessing (`TabularPreprocessor`), fit on the training sample only:
- Non-binary numeric features -> median imputation, then a quantile transform
  to a normal distribution. Several features are heavy-tailed
  (`secs_since_prev_txn` reaches 2.9e7), which raw scaling handles badly.
- Binary 0/1 features pass through unchanged.
- A 0/1 missing-value indicator is added for each column with NaNs in training.
- Categories -> integer index; 0 is reserved for missing or unseen levels.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import QuantileTransformer

from src.modeling import categorical_in


# ----------------------------------------------------------------------------- preprocessing
class TabularPreprocessor:
    def __init__(self, features: list[str], n_quantiles: int = 1000, seed: int = 0):
        self.features = features
        self.cats = categorical_in(features)
        self.nums = [c for c in features if c not in self.cats]
        self.n_quantiles = n_quantiles
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        X = df[self.nums]
        self.binary = [c for c in self.nums if set(pd.unique(X[c].dropna())) <= {0, 1}]
        self.continuous = [c for c in self.nums if c not in self.binary]
        self.with_missing = [c for c in self.nums if X[c].isna().any()]
        self.medians = X[self.continuous].median()
        self.quantile = QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution="normal",
                                            subsample=1_000_000, random_state=self.seed)
        self.quantile.fit(X[self.continuous].fillna(self.medians).to_numpy(np.float64))
        self.levels = {c: df[c].cat.categories for c in self.cats}
        self.cardinalities = [len(self.levels[c]) + 1 for c in self.cats]   # +1: index 0 = missing/unseen
        return self

    @property
    def n_num(self) -> int:
        return len(self.continuous) + len(self.binary) + len(self.with_missing)

    @property
    def num_names(self) -> list[str]:
        return self.continuous + self.binary + [f"missing_{c}" for c in self.with_missing]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cont = self.quantile.transform(df[self.continuous].fillna(self.medians).to_numpy(np.float64))
        binary = df[self.binary].fillna(0).to_numpy(np.float64)
        missing = df[self.with_missing].isna().to_numpy(np.float64)
        x_num = np.hstack([cont, binary, missing]).astype(np.float32)
        x_cat = np.column_stack([
            pd.Categorical(df[c].astype(object), categories=self.levels[c]).codes.astype(np.int64) + 1
            for c in self.cats
        ])                                                  # unseen / NaN: code -1 -> 0
        return x_num, x_cat


# ----------------------------------------------------------------------------- model
class FeatureTokenizer(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], d: int):
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_num, d))
        self.num_bias = nn.Parameter(torch.empty(n_num, d))
        self.cat_embeddings = nn.ModuleList(nn.Embedding(k, d) for k in cardinalities)
        self.cat_bias = nn.Parameter(torch.empty(len(cardinalities), d))
        for p in (self.num_weight, self.num_bias, self.cat_bias):
            nn.init.uniform_(p, -1 / math.sqrt(d), 1 / math.sqrt(d))
        for e in self.cat_embeddings:
            nn.init.uniform_(e.weight, -1 / math.sqrt(d), 1 / math.sqrt(d))

    def forward(self, x_num, x_cat):
        num = x_num.unsqueeze(-1) * self.num_weight + self.num_bias            # (B, n_num, d)
        cat = torch.stack([e(x_cat[:, i]) for i, e in enumerate(self.cat_embeddings)], 1) + self.cat_bias
        return torch.cat([num, cat], 1)


class Block(nn.Module):
    def __init__(self, d, n_heads, attn_dropout, ffn_factor, ffn_dropout, residual_dropout, first):
        super().__init__()
        self.n_heads = n_heads
        self.attn_norm = nn.Identity() if first else nn.LayerNorm(d)   # paper: no norm before the first attention
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.attn_dropout = attn_dropout
        hidden = int(d * ffn_factor)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn_in = nn.Linear(d, 2 * hidden)                          # ReGLU: two halves, one gates the other
        self.ffn_out = nn.Linear(hidden, d)
        self.ffn_drop = nn.Dropout(ffn_dropout)
        self.res_drop = nn.Dropout(residual_dropout)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(B, T, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = Fn.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.res_drop(self.out(a.transpose(1, 2).reshape(B, T, d)))
        h, gate = self.ffn_in(self.ffn_norm(x)).chunk(2, -1)
        x = x + self.res_drop(self.ffn_out(self.ffn_drop(h * Fn.relu(gate))))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_num, cardinalities, d_token=192, n_blocks=3, n_heads=8, attn_dropout=0.2,
                 ffn_factor=4 / 3, ffn_dropout=0.1, residual_dropout=0.0):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cardinalities, d_token)
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.uniform_(self.cls, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        self.blocks = nn.ModuleList(
            Block(d_token, n_heads, attn_dropout, ffn_factor, ffn_dropout, residual_dropout, first=i == 0)
            for i in range(n_blocks))
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        x = self.tokenizer(x_num, x_cat)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, 0]).squeeze(-1)


# ----------------------------------------------------------------------------- training
MODEL_KEYS = ("d_token", "n_blocks", "n_heads", "attn_dropout", "ffn_factor", "ffn_dropout", "residual_dropout")


def to_device(x_num, x_cat, y=None, device="cuda"):
    out = [torch.as_tensor(x_num, device=device), torch.as_tensor(x_cat, device=device)]
    if y is not None:
        out.append(torch.as_tensor(np.asarray(y, dtype=np.float32), device=device))
    return out


@torch.no_grad()
def predict(model, x_num, x_cat, batch_size=16384) -> np.ndarray:
    """Fraud probabilities. Inputs are tensors already on the model's device."""
    model.eval()
    out = []
    for i in range(0, len(x_num), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x_num[i:i + batch_size], x_cat[i:i + batch_size])
        out.append(torch.sigmoid(logits.float()))
    return torch.cat(out).cpu().numpy()


def _save_checkpoint(path, state: dict) -> None:
    """Write to a temp file, then rename: a power cut mid-write leaves the
    previous checkpoint intact instead of a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path):
    if path is None or not Path(path).exists():
        return None
    try:
        return torch.load(path, map_location="cuda", weights_only=False)
    except Exception as e:                      # corrupted file: start over rather than crash
        print(f"  could not read checkpoint {path} ({e}); training from scratch")
        return None


def _lr_at(step, total_steps, warmup_steps, base_lr, schedule):
    """Constant, or linear warm-up then cosine decay to 0."""
    if schedule is None:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def fit_ft(train_tensors, n_num, cardinalities, val_tensors=None, *, epochs=30, patience=4,
           batch_size=1024, lr=1e-4, weight_decay=1e-5, pos_weight=1.0, seed=0, verbose=True,
           max_minutes=None, on_epoch=None, checkpoint=None,
           schedule=None, warmup_epochs=1, ema_decay=None, select_best=True, model_fn=None, **model_params):
    """Train an FT-Transformer (or, with `model_fn`, any model called as model(x_num, x_cat)).

    With `val_tensors` (x_num, x_cat, y) and `select_best=True` it early-stops on
    validation PR-AUC and returns the best epoch's weights. Without `val_tensors`,
    or with `select_best=False`, it trains exactly `epochs` epochs and returns the
    final weights; validation (if given) is then only recorded, never used.
    `on_epoch(epoch, val_pr_auc)` is called after each validated epoch (used for
    Optuna pruning: it may raise to stop training).

    Stabilisers (both off by default, which reproduces the original runs):
    - `schedule="cosine"`: linear warm-up over `warmup_epochs`, then cosine decay
      of the learning rate to 0 at the end of `epochs`.
    - `ema_decay` (e.g. 0.9995): keep an exponential moving average of the weights,
      updated every step. Validation and the returned model use the EMA weights.

    `checkpoint`: a file path. After every epoch the full training state is saved
    there (model, optimizer, RNG states, early-stopping state, history, elapsed
    time). If the file already exists, training resumes from it, e.g. after a power
    cut. If it holds a finished run, the trained model is returned without training.
    The caller must use a different path for each configuration.

    `model_fn(n_num, cardinalities, **model_params)` builds the network; the default
    is `FTTransformer`. `src/mlp.py` uses it to train an MLP with the same loop.

    Returns (model, history), history = list of per-epoch dicts.
    """
    torch.manual_seed(seed)
    x_num, x_cat, y = train_tensors
    model_fn = model_fn or FTTransformer
    model = model_fn(n_num, cardinalities, **model_params).cuda()
    # Paper: no weight decay on the tokenizer / embeddings, norms and biases
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if ("tokenizer" in name or "embedding" in name or "norm" in name or name.endswith("bias") or name == "cls") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    pw = torch.tensor(pos_weight, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    history, best_ap, best_state, best_epoch, bad, elapsed = [], -1.0, None, 0, 0, 0.0
    steps_per_epoch = math.ceil(len(y) / batch_size)
    total_steps, warmup_steps = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch
    ema = None
    if ema_decay is not None:
        ema = model_fn(n_num, cardinalities, **model_params).cuda()
        ema.load_state_dict(model.state_dict())
        ema.requires_grad_(False)
    eval_model = ema if ema is not None else model
    early_stop = val_tensors is not None and select_best

    def finish():
        if early_stop and best_state is not None:
            model.load_state_dict(best_state)
            model.best_epoch = best_epoch
        elif ema is not None:
            model.load_state_dict(ema.state_dict())
            model.best_epoch = len(history)
        else:
            model.best_epoch = len(history)
        return model, history

    ck = _load_checkpoint(checkpoint)
    if ck is not None:
        model.load_state_dict(ck["model"])
        if ema is not None and ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["generator"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        history, best_ap, best_epoch, bad, elapsed = (ck["history"], ck["best_ap"], ck["best_epoch"],
                                                      ck["bad"], ck["elapsed"])
        best_state = ck["best_state"]
        if ck["finished"]:
            if verbose:
                print(f"  loaded finished run from {checkpoint} ({len(history)} epochs)")
            return finish()
        if verbose:
            print(f"  resuming from epoch {len(history)} ({checkpoint})")
        if on_epoch is not None:                # replay finished epochs, e.g. for a new Optuna trial
            for r in history:
                if "val_pr_auc" in r:
                    on_epoch(r["epoch"], r["val_pr_auc"])

    def save(finished):
        if checkpoint is not None:
            _save_checkpoint(checkpoint, {
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "generator": gen.get_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "history": history, "best_ap": best_ap, "best_epoch": best_epoch, "bad": bad,
                "best_state": best_state, "elapsed": elapsed, "finished": finished})

    params = list(model.parameters())
    ema_params = list(ema.parameters()) if ema is not None else None
    stop = False
    for epoch in range(len(history) + 1, epochs + 1):
        model.train()
        t0 = time.time()
        step = (epoch - 1) * steps_per_epoch
        perm = torch.randperm(len(y), device="cuda", generator=gen)
        total = torch.zeros((), device="cuda")
        for i in range(0, len(y), batch_size):
            idx = perm[i:i + batch_size]
            if schedule is not None:
                cur_lr = _lr_at(step, total_steps, warmup_steps, lr, schedule)
                for g in opt.param_groups:
                    g["lr"] = cur_lr
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x_num[idx], x_cat[idx])
            loss = Fn.binary_cross_entropy_with_logits(logits.float(), y[idx], pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema is not None:                     # usual EMA warm-up: short memory in the first steps
                d = min(ema_decay, (1 + step) / (10 + step))
                with torch.no_grad():
                    torch._foreach_lerp_(ema_params, params, 1 - d)
            step += 1
            total += loss.detach() * len(idx)       # no .item(): avoids a GPU sync every step
        rec = {"epoch": epoch, "train_loss": total.item() / len(y), "secs": time.time() - t0}
        if val_tensors is not None:
            p = predict(eval_model, val_tensors[0], val_tensors[1])
            rec["val_pr_auc"] = average_precision_score(val_tensors[2].cpu().numpy(), p)
            if rec["val_pr_auc"] > best_ap:
                best_ap, best_epoch, bad = rec["val_pr_auc"], epoch, 0
                if early_stop:
                    best_state = {k: v.detach().clone() for k, v in eval_model.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        elapsed += time.time() - t0
        if verbose:
            print("  " + " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
        stop = ((early_stop and bad >= patience)
                or (max_minutes is not None and elapsed / 60 > max_minutes))
        save(finished=stop or epoch == epochs)
        if on_epoch is not None and "val_pr_auc" in rec:
            on_epoch(epoch, rec["val_pr_auc"])
        if stop:
            break
    if not stop:
        save(finished=True)                         # loop ran to `epochs` (or had nothing left to run)
    return finish()


def checkpoint_name(tag: str, config: dict) -> str:
    """Stable file name for one training configuration, e.g. 'trial_3f9a1c2e.pt'."""
    key = json.dumps(config, sort_keys=True, default=str)
    return f"{tag}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.pt"
