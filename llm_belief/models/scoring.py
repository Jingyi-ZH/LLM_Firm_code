"""Scoring models and pairwise training utilities."""

from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


class MLPAttentionScore(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: tuple = (256, 128),
        dropout: float = 0.1,
        n_heads: int = 4,
        d_model: int = 64,
    ):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.feat_emb = nn.Parameter(torch.randn(d_in, d_model))

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.pool = nn.AdaptiveAvgPool1d(1)

        mlp_layers = []
        dims = [d_model] + list(hidden)
        for a, b in zip(dims[:-1], dims[1:]):
            mlp_layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        mlp_layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*mlp_layers)

    def forward(self, x, return_attn: bool = False):
        t = x.unsqueeze(-1)
        tokens = self.value_proj(t) + self.feat_emb

        attn_out, attn_w = self.attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=False
        )
        z = self.norm1(tokens + attn_out)
        z = self.norm2(z + self.ffn(z))

        z = z.transpose(1, 2)
        z = self.pool(z).squeeze(-1)

        score = self.net(z).squeeze(-1)
        if return_attn:
            return score, attn_w
        return score


class MLPScorer(nn.Module):
    def __init__(self, d_in: int, hidden: tuple = (256, 128), dropout: float = 0.1):
        super().__init__()
        layers = []
        dims = [d_in] + list(hidden)
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dims[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pairwise_logistic_loss(si, sj, y_ij):
    margin = y_ij * (si - sj)
    return F.softplus(-margin).mean()


def train_pairwise(
    model,
    train_loader,
    valid_loader=None,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_val = None, float("inf")
    patience, bad = 3, 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for Xi, Xj, y in train_loader:
            Xi, Xj, y = Xi.to(device), Xj.to(device), y.to(device)
            si, sj = model(Xi), model(Xj)
            loss = pairwise_logistic_loss(si, sj, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            opt.step()
            running += loss.item() * y.size(0)

        train_loss = running / len(train_loader.dataset)
        val_loss = None
        if valid_loader is not None:
            model.eval()
            vrun = 0.0
            with torch.no_grad():
                for Xi, Xj, y in valid_loader:
                    Xi, Xj, y = Xi.to(device), Xj.to(device), y.to(device)
                    si, sj = model(Xi), model(Xj)
                    vrun += pairwise_logistic_loss(si, sj, y).item() * y.size(0)
            val_loss = vrun / len(valid_loader.dataset)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        msg = f"[Epoch {ep:02d}] train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def calibrate_temperature(model, valid_loader):
    device = next(model.parameters()).device
    T = torch.tensor(1.0, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        losses = []
        for Xi, Xj, y in valid_loader:
            Xi, Xj, y = Xi.to(device), Xj.to(device), y.to(device)
            si, sj = model(Xi), model(Xj)
            margin = y * ((si - sj) / T.clamp_min(1e-3))
            loss = F.softplus(-margin).mean()
            losses.append(loss)
        loss = torch.stack(losses).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        Tval = float(T.clamp_min(1e-3).item())
    return Tval


def score_profiles(model, X, T: float = 1.0):
    device = next(model.parameters()).device
    with torch.no_grad():
        scores = model(X.to(device)).cpu().numpy().reshape(-1)
    scores_stable = (scores - scores.max()) / max(T, 1e-6)
    w = np.exp(scores_stable)
    probs = w / w.sum()
    return scores, probs
