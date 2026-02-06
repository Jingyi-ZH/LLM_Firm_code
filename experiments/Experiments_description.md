# Experiments Description

This document describes how to reproduce **Experiment 1**: goal, inputs, settings, commands, and outputs.

## Experiment 1 — Position real profiles among 100 representative hypothetical profiles

### 1) Goal (what we are measuring)
The goal is to **position real iPhone profiles (e.g., iPhone 16) among a set of 100 representative hypothetical profiles** using a direct pairwise-comparison method.

We operationalize “positioning” as follows:
- Fix a shared set of `n_makeup=5000` hypothetical profiles (sampled once and reused).
- For each target profile (a real iPhone profile, or one of the 100 representatives), run **fixreal** comparisons against the *same* fixed set of 5000 hypothetical profiles.
- Use the resulting win rates / logprobs-derived probabilities to position (rank) the real iPhone profiles relative to the representative set.

### 2) The 100 representative profiles (AlgDesign, criterion = I)
The file `experiments/alternatives/design100.csv` contains 100 “representative” hypothetical profiles selected using the R package `AlgDesign` with the **I** criterion (I-optimality).

High-level intuition:
- The full design space (all attribute combinations) is large.
- `AlgDesign` chooses a smaller subset that is maximally informative for prediction under a specified model, rather than selecting profiles uniformly at random.
- With **I-optimality**, the design is chosen to minimize the *average* prediction variance over the candidate/design space (often described as minimizing the integrated/average variance of predicted responses). Practically, this yields a set of profiles that “covers” the space well for prediction and comparison.

### 3) Fixed makeup sample (n_makeup = 5000)
Fixreal samples `n_makeup` `profile_ids` from the first `collection.fixreal_sample_limit` rows of `data/profiles_shuffled.csv`.
- The sampled ids are cached as `data/sample{n_makeup}_profile_ids.npy` (e.g., `data/sample5000_profile_ids.npy`).
- If the cache file exists, it is reused (no re-sampling), ensuring all targets are compared against the same fixed set.

### 4) Settings
- `--logprobs on`: enables logprobs logging; the output CSV adds `prob_chosen` and `prob_nochosen`.
- The logprobs model (`gpt-4.1-nano`)/temperature (`0`)/etc. are configured in `config/config.yaml` under `openai.logprobs.*`

Note:
- When `--logprobs on`, `--reasoning-effort` does not take effect (current implementation does not pass reasoning effort in the logprobs API call path).

### 5) Commands

Run fixreal for the 100 representative profiles (one output per row):
```bash
llm-collect --experiment fixreal --real-profile experiments/alternatives/design100.csv \
  --n-makeup 5000 --output experiments/experiment_1/ --logprobs on
```

Run fixreal for the real iPhone profiles (one output per row):
```bash
llm-collect --experiment fixreal --real-profile experiments/alternatives/real_profiles.csv \
  --n-makeup 5000 --output experiments/experiment_1/ --logprobs on
```

### 6) Outputs

Each row (each `real_profile_id`) produces one output file:
- `{real_profile_id_with_underscores}_fixreal{n_makeup}.csv`
- Example: `iPhone 16` → `iPhone_16_fixreal5000.csv`

With `--output experiments/experiment_1/`, outputs are written to:
- `experiments/experiment_1/`
