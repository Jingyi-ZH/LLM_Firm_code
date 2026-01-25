"""Adaptive Lasso pairwise model utilities."""

from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMap(nn.Module):
    def __init__(self, in_dim, include_interaction: bool = True, poly_degree: int = 1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.include_interaction = bool(include_interaction)
        self.poly_degree = max(1, int(poly_degree))

        pairs = []
        if self.include_interaction:
            for i in range(self.in_dim):
                for j in range(i + 1, self.in_dim):
                    pairs.append((i, j))

        if pairs:
            self.register_buffer(
                "pair_i", torch.tensor([p[0] for p in pairs], dtype=torch.long)
            )
            self.register_buffer(
                "pair_j", torch.tensor([p[1] for p in pairs], dtype=torch.long)
            )
        else:
            self.register_buffer("pair_i", torch.empty(0, dtype=torch.long))
            self.register_buffer("pair_j", torch.empty(0, dtype=torch.long))
        self.num_pairs = len(pairs)

    def out_dim(self):
        poly_extra = (self.poly_degree - 1) * self.in_dim
        return self.in_dim + self.num_pairs + poly_extra

    def forward(self, X):
        outs = [X]
        if self.num_pairs > 0:
            inter = X[:, self.pair_i] * X[:, self.pair_j]
            outs.append(inter)
        for k in range(2, self.poly_degree + 1):
            outs.append(X.pow(k))
        return torch.cat(outs, dim=1)


class LinearInteractionModel(nn.Module):
    def __init__(self, in_dim, include_interaction: bool = True, poly_degree: int = 1):
        super().__init__()
        self.mapper = FeatureMap(
            in_dim, include_interaction=include_interaction, poly_degree=poly_degree
        )
        self.scorer = nn.Linear(self.mapper.out_dim(), 1, bias=True)

    def forward(self, X):
        Z = self.mapper(X)
        s = self.scorer(Z)
        return s.squeeze(-1)


@torch.no_grad()
def list_nonzero_terms(model, threshold: float = 1e-8):
    coef = model.scorer.weight.detach().cpu().flatten()
    in_dim = model.mapper.in_dim
    num_pairs = model.mapper.num_pairs
    poly_degree = model.mapper.poly_degree

    offset_main = 0
    offset_pairs = offset_main + in_dim
    offset_poly_base = offset_pairs + num_pairs

    main = []
    for i in range(in_dim):
        c = float(coef[offset_main + i].item())
        if abs(c) > threshold:
            main.append((i, c))

    interactions = []
    if num_pairs > 0:
        pair_i = model.mapper.pair_i.detach().cpu().tolist()
        pair_j = model.mapper.pair_j.detach().cpu().tolist()
        for t in range(num_pairs):
            c = float(coef[offset_pairs + t].item())
            if abs(c) > threshold:
                interactions.append(((pair_i[t], pair_j[t]), c))

    poly = []
    for k in range(2, poly_degree + 1):
        start = offset_poly_base + (k - 2) * in_dim
        for i in range(in_dim):
            c = float(coef[start + i].item())
            if abs(c) > threshold:
                poly.append(((i,) * k, c))

    return {"main": main, "interactions": interactions, "poly": poly}


def pairwise_logistic_loss(si, sj, y_ij):
    margin = y_ij * (si - sj)
    return F.softplus(-margin).mean()


def evaluate_pairwise_loss(model, valid_loader, device):
    model.eval()
    vrun = 0.0
    with torch.no_grad():
        for Xi, Xj, y in valid_loader:
            Xi, Xj, y = Xi.to(device), Xj.to(device), y.to(device)
            si, sj = model(Xi), model(Xj)
            vrun += pairwise_logistic_loss(si, sj, y).item() * y.size(0)
    return vrun / len(valid_loader.dataset)


def train_pairwise(
    model,
    train_loader,
    valid_loader=None,
    lr=1e-3,
    weight_decay=1e-5,
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
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
            val_loss = evaluate_pairwise_loss(model, valid_loader, device=device)
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


def compute_adaptive_weights_from_model(model, eps=1e-4, gamma=1.0):
    coef = model.scorer.weight.detach().abs().clamp_min(eps)
    return 1.0 / (coef.pow(gamma))


def make_weighted_L1_reg(model, w, lam, device=None):
    if device is None:
        device = next(model.parameters()).device
    w = w.to(device)

    def reg_fn(_model):
        return lam * (w * _model.scorer.weight.abs()).sum()

    return reg_fn


def train_pairwise_adalasso(
    model: nn.Module,
    train_loader,
    valid_loader=None,
    *,
    lr=2e-3,
    weight_decay=0.0,
    warmup_epochs=5,
    finetune_epochs=20,
    patience=3,
    gamma=1.0,
    eps=1e-4,
    lambda_grid: Optional[Sequence[float]] = None,
    lambda_scale: float = 1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)

    print("== Warmup without L1 ==")
    model = train_pairwise(
        model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=lr,
        weight_decay=weight_decay,
        epochs=warmup_epochs,
        device=device,
    )
    warm_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    w = compute_adaptive_weights_from_model(model, eps=eps, gamma=gamma)

    if lambda_grid is None:
        lambda_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    lambda_grid = [lam * lambda_scale for lam in lambda_grid]

    best_val = float("inf")
    best_state = None
    best_lambda = None

    for lam in lambda_grid:
        print(f"== Try lambda={lam:.2e} ==")
        model.load_state_dict(warm_state)
        model = model.to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        reg_fn = make_weighted_L1_reg(model, w, lam, device=device)

        bad = 0
        last_val = float("inf")

        for ep in range(1, finetune_epochs + 1):
            model.train()
            running = 0.0
            for Xi, Xj, y in train_loader:
                Xi, Xj, y = Xi.to(device), Xj.to(device), y.to(device)
                si, sj = model(Xi), model(Xj)
                margin = y * (si - sj)
                loss = F.softplus(-margin).mean() + reg_fn(model)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                opt.step()

                running += loss.item() * y.size(0)

            train_loss = running / len(train_loader.dataset)
            val_loss = (
                evaluate_pairwise_loss(model, valid_loader, device=device)
                if valid_loader is not None
                else train_loss
            )

            improved = val_loss < last_val - 1e-6
            last_val = val_loss

            print(
                f"   [lambda={lam:.2e} ep={ep:02d}] train={train_loss:.4f}  val={val_loss:.4f}"
                + ("  *" if improved else "")
            )

            if improved:
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if last_val < best_val:
            best_val = last_val
            best_lambda = lam
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "best_lambda": best_lambda,
        "best_val": best_val,
        "adaptive_weights": w.detach().cpu(),
        "coef": model.scorer.weight.detach().cpu().numpy(),
        "bias": float(model.scorer.bias.detach().cpu().item()),
        "feature_dim": model.scorer.weight.shape[1],
        "num_main_effects": model.mapper.in_dim,
        "num_interactions": model.mapper.num_pairs,
    }
    print(f"== Best lambda: {best_lambda}  val={best_val:.6f}")
    return model, info
