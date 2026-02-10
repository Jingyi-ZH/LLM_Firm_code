# Experiments Description

This document describes how to reproduce **Experiment 1**: goal, inputs, settings, commands, and outputs.

## Experiment 1 — Position real profiles among 100 representative hypothetical profiles

### Results — Real iPhone profile rankings (out of 108 profiles)

> | profile_id | rank_p_win_mean | rank_p_win_median | rank_win_rate |
> |---|---|---|---|
> | iPhone 17 | 6 | 8 | 7 |
> | iPhone 17 Air | 11 | 15 | 9 |
> | iPhone 17 Pro | 21 | 41 | 18 |
> | iPhone 17 Pro Max | 60 | 67 | 22 |
> | iPhone 16 | 17 | 10 | 36 |
> | iPhone 16 Plus | 30 | 30 | 45 |
> | iPhone 16 Pro | 49 | 55 | 52 |
> | iPhone 16 Pro Max | 74 | 75 | 63 |

### Conclusions

> 1. Most attributes have a relatively **uniform** influence on win_rate / prob, but the model clearly **favors high Geekbench scores** and **low price**. These two attributes appear to be the strongest drivers of preference.
>
> 2. Top-ranked profiles tend to have **very low device thickness** — this likely explains why **iPhone 17 Air** (5.64 mm, the thinnest by far) ranks surprisingly high (rank 9 by win_rate, rank 11 by p_win_mean) despite not being equipped with an ultrawide camera.
>
> 3. **High Geekbench scores favor the entire iPhone 17 series**: all four iPhone 17 models (Geekbench 9191–9776) rank above their iPhone 16 counterparts (Geekbench 8205–8581) across all three metrics (win_rate, p_win_mean, p_win_median). The generational Geekbench gap (~600–1200 points) appears to be a major differentiator.

---

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
  --n-makeup 5000 --output output/experiment_1/ --logprobs on
```

Run fixreal for the real iPhone profiles (one output per row):
```bash
llm-collect --experiment fixreal --real-profile experiments/alternatives/real_profiles.csv \
  --n-makeup 5000 --output output/experiment_1/ --logprobs on
```

### 6) Outputs

Each row (each `real_profile_id`) produces one output file:
- `{real_profile_id_with_underscores}_fixreal{n_makeup}.csv`
- Example: `iPhone 16` → `iPhone_16_fixreal5000.csv`

With `--output output/experiment_1/`, outputs are written to:
- `output/experiment_1/`

### 7) Sanity check — GPT-4.1-nano knowledge cutoff issue

OpenAI's official documentation states that GPT-4.1-nano has a knowledge cutoff of **June 2024**. However, in our sanity checks, the model self-reported inconsistent cutoff dates:
- Responded **"October 2023"** in two out of three runs
- Responded **"September 2021"** in one run

This is completely misaligned with the official cutoff (June 2024) and indicates that **GPT-4.1-nano's self-reported knowledge cutoff is unreliable**. Furthermore, the model's response about iPhone 15 screen sizes was incomplete (missing Pro and Pro Max models) and used hedging language like "expected or latest if announced", suggesting its factual knowledge may not fully cover the period up to its official cutoff date.

See `sanity_check_summary.xlsx` for the full record of all sanity check results.

