#!/usr/bin/env python3
"""CLI for model training.

This script trains scoring models on pairwise comparison outputs.

Usage:
    python scripts/run_training.py --model logistic --input-glob "output/*_*.csv"
    python scripts/run_training.py --model mlp --inputs output/0_1000.csv,output/1000_2000.csv
    python scripts/run_training.py --model xgboost --input-glob "output/*_*.csv"
"""

import argparse
import sys
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

from config import get_config
from llm_belief.utils.paths import get_data_path, get_models_path
from llm_belief.preprocessing import preprocess, PairwiseDataset
from llm_belief.models.scoring import (
    LogisticRegression,
    MLPScorer,
    MLPAttentionScore,
    train_pairwise,
)
from llm_belief.models.adalasso import LinearInteractionModel, train_pairwise_adalasso
from llm_belief.models.xgboost_model import train_xgb_pairwise


def main():
    """Main entry point for training CLI."""
    parser = argparse.ArgumentParser(
        description="LLM Belief Elicitation Model Training",
    )

    parser.add_argument(
        "--model",
        choices=["logistic", "mlp", "mlp_attention", "adalasso", "xgboost"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        help="Comma-separated list of output CSV files",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="output/*.csv",
        help="Glob pattern for output CSV files (used if --inputs not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed override",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device (e.g., cpu, cuda, mps)",
    )
    parser.add_argument(
        "--include-interactions",
        action="store_true",
        help="Include interaction terms for adaptive lasso",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=1,
        help="Polynomial degree for adaptive lasso (default: 1)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Warmup epochs for adaptive lasso",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=20,
        help="Finetune epochs for adaptive lasso",
    )
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=1.0,
        help="Lambda scale for adaptive lasso",
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=300,
        help="Boosting rounds for xgboost",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=30,
        help="Early stopping rounds for xgboost",
    )

    args = parser.parse_args()

    cfg = get_config()
    seed = args.seed if args.seed is not None else cfg.get("project", "random_seed", default=2025)
    rng = np.random.default_rng(seed)

    if args.inputs:
        input_files = [p.strip() for p in args.inputs.split(",") if p.strip()]
    else:
        input_files = glob.glob(args.input_glob)

    if not input_files:
        raise FileNotFoundError("No input CSV files found for training.")

    df_list = []
    for path in input_files:
        df_list.append(pd.read_csv(path))
    pairs_df = pd.concat(df_list, ignore_index=True)

    if "pair_id" not in pairs_df.columns or "profile_id" not in pairs_df.columns:
        raise ValueError("Input CSVs must contain 'pair_id' and 'profile_id' columns.")

    pairs_df["pair_id"] = pd.to_numeric(pairs_df["pair_id"], errors="coerce")
    pairs_df["profile_id"] = pd.to_numeric(pairs_df["profile_id"], errors="coerce")
    pairs_df = pairs_df.dropna(subset=["pair_id", "profile_id"])
    pairs_df["pair_id"] = pairs_df["pair_id"].astype(int)
    pairs_df["profile_id"] = pairs_df["profile_id"].astype(int)

    profiles_file = cfg.get("collection", "profiles_file")
    profiles_df = pd.read_csv(get_data_path(profiles_file))
    X = preprocess(profiles_df)

    pair_ids = pairs_df["pair_id"].to_numpy()
    chosen_ids = pairs_df["profile_id"].to_numpy()
    valid_mask = (chosen_ids == 2 * pair_ids) | (chosen_ids == 2 * pair_ids + 1)
    pairs_df = pairs_df[valid_mask]
    pair_ids = pairs_df["pair_id"].to_numpy()
    chosen_ids = pairs_df["profile_id"].to_numpy()

    idx_i = 2 * pair_ids
    idx_j = 2 * pair_ids + 1
    Xi = X[idx_i]
    Xj = X[idx_j]

    if "profile_id_nochoose" in pairs_df.columns:
        nochoose = pd.to_numeric(pairs_df["profile_id_nochoose"], errors="coerce")
        nochoose = nochoose.fillna(-1).astype(int)
        y_vals = nochoose - pairs_df["profile_id"].astype(int)
        y = torch.from_numpy(y_vals.to_numpy()).float()
    else:
        y = torch.from_numpy(np.where(chosen_ids % 2 == 0, 1.0, -1.0)).float()

    num_pairs = Xi.shape[0]
    perm = rng.permutation(num_pairs)
    split = int((1.0 - args.val_ratio) * num_pairs)
    train_idx = perm[:split]
    val_idx = perm[split:]

    ds_tr = PairwiseDataset(Xi[train_idx], Xj[train_idx], y[train_idx])
    ds_va = PairwiseDataset(Xi[val_idx], Xj[val_idx], y[val_idx])

    batch_size = args.batch_size or cfg.get("training", "batch_size", default=64)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    lr = args.lr or cfg.get("training", "learning_rate", default=1e-3)
    epochs = args.epochs or cfg.get("training", "epochs", default=50)
    device = args.device or cfg.get("training", "device", default="auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d_in = X.shape[1]
    output_path = args.output
    if not output_path:
        suffix = "pt"
        if args.model == "xgboost":
            suffix = "json"
        output_path = str(get_models_path(f"{args.model}.{suffix}"))

    if args.model == "logistic":
        model = LogisticRegression(d_in=d_in)
        model = train_pairwise(model, dl_tr, dl_va, lr=lr, epochs=epochs, device=device)
        torch.save({"state_dict": model.state_dict(), "d_in": d_in}, output_path)
    elif args.model == "mlp":
        model = MLPScorer(d_in=d_in)
        model = train_pairwise(model, dl_tr, dl_va, lr=lr, epochs=epochs, device=device)
        torch.save({"state_dict": model.state_dict(), "d_in": d_in}, output_path)
    elif args.model == "mlp_attention":
        model = MLPAttentionScore(d_in=d_in)
        model = train_pairwise(model, dl_tr, dl_va, lr=lr, epochs=epochs, device=device)
        torch.save({"state_dict": model.state_dict(), "d_in": d_in}, output_path)
    elif args.model == "adalasso":
        model = LinearInteractionModel(
            in_dim=d_in,
            include_interaction=args.include_interactions,
            poly_degree=args.poly_degree,
        )
        model, info = train_pairwise_adalasso(
            model,
            train_loader=dl_tr,
            valid_loader=dl_va,
            lr=lr,
            warmup_epochs=args.warmup_epochs,
            finetune_epochs=args.finetune_epochs,
            lambda_scale=args.lambda_scale,
            device=device,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "d_in": d_in,
                "info": info,
            },
            output_path,
        )
    elif args.model == "xgboost":
        booster = train_xgb_pairwise(
            Xi[train_idx],
            Xj[train_idx],
            y[train_idx],
            Xi[val_idx],
            Xj[val_idx],
            y[val_idx],
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        booster.save_model(output_path)

    print(f"Saved model to: {output_path}")


if __name__ == "__main__":
    main()
