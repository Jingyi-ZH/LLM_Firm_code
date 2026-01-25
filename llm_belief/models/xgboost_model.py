"""XGBoost pairwise ranking utilities."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb


def pairwise_logistic_loss(si, sj, y_ij):
    margin = y_ij * (si - sj)
    return F.softplus(-margin).mean()


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


def score_profiles(model, X, T=1.0):
    device = next(model.parameters()).device
    with torch.no_grad():
        scores = model(X.to(device)).cpu().numpy().reshape(-1)
    scores_stable = (scores - scores.max()) / max(T, 1e-6)
    w = np.exp(scores_stable)
    probs = w / w.sum()
    return scores, probs


def build_xgb_rank_dmatrix(
    Xi: torch.Tensor, Xj: torch.Tensor, y: torch.Tensor
) -> xgb.DMatrix:
    Xi_np = Xi.detach().cpu().numpy()
    Xj_np = Xj.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().astype(np.int32)

    X_stacked = np.vstack([Xi_np, Xj_np])

    rel_i = (y_np == +1).astype(np.int32)
    rel_j = (y_np == -1).astype(np.int32)
    y_stacked = np.concatenate([rel_i, rel_j]).astype(np.float32)

    group = np.full(shape=(Xi_np.shape[0],), fill_value=2, dtype=np.int32)

    dtrain = xgb.DMatrix(X_stacked, label=y_stacked)
    dtrain.set_group(group)
    return dtrain


def train_xgb_pairwise(
    Xi_tr: torch.Tensor,
    Xj_tr: torch.Tensor,
    y_tr: torch.Tensor,
    Xi_va: torch.Tensor | None = None,
    Xj_va: torch.Tensor | None = None,
    y_va: torch.Tensor | None = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
    params: dict | None = None,
):
    if params is None:
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@2",
            "tree_method": "hist",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
        }

    dtrain = build_xgb_rank_dmatrix(Xi_tr, Xj_tr, y_tr)
    evals = [(dtrain, "train")]
    if Xi_va is not None and Xj_va is not None and y_va is not None:
        dvalid = build_xgb_rank_dmatrix(Xi_va, Xj_va, y_va)
        evals.append((dvalid, "valid"))
    else:
        dvalid = None

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if dvalid is not None else None,
        verbose_eval=50,
    )
    return booster


class XGBScorerTorch(nn.Module):
    def __init__(self, booster: xgb.Booster, d_in: int, device: str | None = None):
        super().__init__()
        self.booster = booster
        self.d_in = d_in
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        dmat = xgb.DMatrix(x_np)
        best_iter = (
            self.booster.best_iteration + 1
            if hasattr(self.booster, "best_iteration")
            and self.booster.best_iteration is not None
            else 0
        )
        pred = self.booster.predict(dmat, iteration_range=(0, best_iter))
        return torch.from_numpy(pred).to(self._dummy.device).float()
