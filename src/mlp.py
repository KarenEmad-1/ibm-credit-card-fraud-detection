"""Feed-forward neural network (MLP) for tabular data.

The MLP baseline of Gorishniy et al. (2021), the paper the FT-Transformer comes from:

    [numeric features, category embeddings] -> n_blocks x (Linear -> ReLU -> Dropout) -> Linear

It uses the FT-Transformer's preprocessing (`TabularPreprocessor`: quantile transform
for continuous features, missing-value indicators, an index per category level) and
its training loop (`fit_ft` with `model_fn=MLP`: AdamW, checkpoints every epoch,
optional cosine schedule and EMA). The only difference between the two models is
therefore the network: no attention, no feature tokens, each category is one learned
vector and everything is concatenated into one input vector.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.ft_transformer import fit_ft

MLP_KEYS = ("d_embedding", "d_hidden", "n_blocks", "dropout")


class MLP(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], d_embedding: int = 16, d_hidden: int = 256,
                 n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(k, d_embedding) for k in cardinalities)
        for e in self.embeddings:
            nn.init.uniform_(e.weight, -1 / math.sqrt(d_embedding), 1 / math.sqrt(d_embedding))
        d_in = n_num + d_embedding * len(cardinalities)
        layers = []
        for i in range(n_blocks):
            layers += [nn.Linear(d_in if i == 0 else d_hidden, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(d_hidden if n_blocks else d_in, 1)

    def forward(self, x_num, x_cat):
        cat = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat([x_num, *cat], 1)
        return self.head(self.blocks(x)).squeeze(-1)


def fit_mlp(train_tensors, n_num, cardinalities, val_tensors=None, **kw):
    """`fit_ft` with the MLP as the network. Takes the same arguments; the model
    arguments are `MLP_KEYS`."""
    return fit_ft(train_tensors, n_num, cardinalities, val_tensors, model_fn=MLP, **kw)
