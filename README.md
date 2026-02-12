# LLM Belief Elicitation

A research project for eliciting iPhone specification beliefs from Large Language Models using pairwise conjoint comparisons.

## Overview

This project implements a complete pipeline for:
1. **Generate Hypothetical Profiles**: Creating iPhone specification profiles for comparison
2. **Data Collection**: Eliciting beliefs from LLMs through pairwise comparisons
3. **Model Training**: Training scoring models (Logistic Regression, MLP, XGBoost, Adaptive Lasso)
4. **Interpretability Analysis**: Integrated Gradients, Partial Dependence Plots, Probability Field Visualization
5. **Data Collection with External Knowledge**: RAG-enhanced/Context-injection belief elicitation

Detailed pipeline (current workflow):
1. Generate hypothetical profiles and save in `data/profiles_shuffled.csv`
2. Collect pairwise comparisons from 20,000 hypothetical profiles
3. Run fixreal comparisons (5,000 hypothetical profiles vs real profiles) to rank real profiles
4. Train scoring models on step 2 outputs and write `data/scored_profiles_shuffled.csv`
5. Visualize results (2D/3D probability fields, PDPs) and generate summary tables
6. Collect external-knowledge data (RAG/context injection) using fixreal mode

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd LLM_Firm_code

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

All other configuration options are in `config/config.yaml`.
Note: `logprobs=on` only works with non-reasoning models.

### 3. Generate Hypothetical Profiles

```bash
python scripts/run_preprocessing.py --task generate-profiles

# With custom seed
python scripts/run_preprocessing.py --task generate-profiles --seed 42

# With custom output filename
python scripts/run_preprocessing.py --task generate-profiles --output custom_profiles.csv
```

This creates `data/profiles_shuffled.csv` from `config/config.yaml`.

### 4. Run Data Collection

Use the Collector CLI (`llm-collect` or `python scripts/run_collection.py`). See **Collector (run_collection.py / llm-collect)** below for the full parameter list, defaults, and examples.

### 5. Run Model Training

```bash
# Train a logistic regression scorer
python scripts/run_training.py --model logistic --input-glob "output/*_*.csv"

# Train an MLP scorer
python scripts/run_training.py --model mlp --input-glob "output/*_*.csv"

# Train XGBoost ranker
python scripts/run_training.py --model xgboost --input-glob "output/*_*.csv"
```

By default, models are saved under `models/`. Training expects basic pairwise
outputs with `pair_id` and `profile_id` columns (e.g., `output/0_1000.csv`).

### 6. Run Analysis

Open the Jupyter notebooks in the `notebooks/` directory:

- **Model Training**: `notebooks/model_training/`
- **Dimensionality Reduction**: `notebooks/dimensionality_reduction/probability_all_models_FA.ipynb`
- **Interpretability**: `notebooks/interpretability/integrated_gradient.ipynb`, `notebooks/interpretability/partial_dependent_plot.ipynb`

Unified visualization API (new):

```python
from llm_belief.analysis import visualize_probability_distribution

result = visualize_probability_distribution(
    model,
    X_full,
    T=1.0,
    method="pca",  # pca | tsne | fa | kpca
    dims=3,        # 3 for surface, 2 for heatmap
    fig_name="prob_field_pca_3d.png",
)
```

## Project Structure

```
LLM_Firm_code/
├── config/                 # Configuration management
│   ├── config.yaml         # Main configuration file
│   └── __init__.py         # Config loading module
├── llm_belief/             # Core Python package
│   ├── data_collection/    # LLM API interaction
│   ├── models/             # Scoring models
│   ├── analysis/           # Analysis modules
│   ├── preprocessing/      # Data preprocessing
│   └── utils/              # Utilities (paths, attributes)
├── scripts/                # CLI entry points
│   ├── run_collection.py   # Data collection CLI
│   ├── run_preprocessing.py# Preprocessing CLI
│   ├── run_training.py     # Model training CLI
│   └── run_analysis.py     # Analysis CLI
├── notebooks/              # Curated analysis notebooks
│   ├── model_training/     # Model training notebooks
│   ├── dimensionality_reduction/
│   ├── interpretability/
│   ├── method_compare/
│   └── auxiliary/
├── RAG_langchain/          # RAG scripts and utilities
│   └── rag_langchain.ipynb
├── data/                   # Data files
│   ├── profiles_shuffled.csv
│   ├── real_profiles.csv   # Legacy (real profiles now in config/config.yaml)
│   ├── scored_profiles_shuffled.csv
│   ├── re16.txt            # Level 1 knowledge injected into prompts
│   └── re16ru17.txt         # Level 2 knowledge injected into prompts
├── output/                 # LLM response outputs
├── plot/                   # Generated visualizations
├── logs/                   # Execution logs
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Modern Python packaging
└── README.md               # This file
```

## Configuration

All configuration is centralized in `config/config.yaml`:

### OpenAI Settings
```yaml
openai:
  model: "gpt-5-nano"
  temperature: 1.0
  reasoning_effort: "medium"
  api_key_env_var: "OPENAI_API_KEY"
```

The runtime uses these values directly (model, temperature, reasoning effort, random seed). API calls use the OpenAI **Responses** API, and all defaults should be defined in `config/config.yaml` rather than hardcoded in scripts.

### Training Settings
```yaml
training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 50
  device: "auto"  # auto, cuda, cpu, mps
```

### Path Configuration
```yaml
paths:
  data_dir: "data"
  output_dir: "output"
  logs_dir: "logs"
  plot_dir: "plot"
```

## Hypothetical Profile Generation

Hypothetical (makeup) profiles are defined by `attributes` in `config/config.yaml`.
To generate the full shuffled profile set:

```bash
python scripts/run_preprocessing.py --task generate-profiles
```

Or use the library directly:

```python
from llm_belief.preprocessing import ProfileGenerator

generator = ProfileGenerator(seed=2025)
profiles = generator.generate()           # Returns list of profile dicts
output_path = generator.generate_csv()    # Generates and saves to CSV
```

This writes `data/profiles_shuffled.csv` using `llm_belief.utils.attributes.generate_all_profiles(...)`.

Real profiles are defined in `config/config.yaml` under `real_profiles`. Prompts assume 256 GB storage and black color as a consistent baseline.

## Experiments

This project supports the following experiment modes:
- `basic`: pairwise comparisons between hypothetical profiles.
- `fixreal`: pairwise comparisons between a *real* profile and sampled hypothetical (“makeup”) profiles.
- `top`: comparisons between a real profile and the top-scored hypothetical profiles.
- `context`: fixreal with injected context text (system message).
- `rag`: fixreal with RAG context via the `RAG_langchain/` pipeline.
- `rag-faiss`: fixreal with a custom FAISS-based retriever (local index + metadata).

See **Collector (run_collection.py / llm-collect)** for how to run each mode and where outputs are written.

## Collector (run_collection.py / llm-collect)

Entry points:
- `python scripts/run_collection.py ...`
- `llm-collect ...` (same CLI, installed by `pip install -e .`)

### Parameters (mandatory vs optional)

| Parameter | Required? | Applies to | Meaning / expected input |
|---|---:|---|---|
| `--experiment` | Yes | all | One of `basic\|fixreal\|top\|context\|rag\|rag-faiss`. |
| `--start` | Yes* | `basic` | Start index (inclusive) of pair range. |
| `--end` | Yes* | `basic` | End index (exclusive) of pair range. |
| `--real-profile` | Yes* | all except `basic` | **Usually**: a real profile id from `config/config.yaml` → `real_profiles`. **Special (fixreal only)**: you may pass a `.csv` path to run fixreal once per row (see below). |
| `--context` | Yes* | `context` | Context text file path (relative to project `data/` or absolute). |
| `--n-makeup` | Optional | `fixreal`, `rag`, `rag-faiss` | Number of sampled makeup profiles. Default: `config/config.yaml` → `collection.default_n_makeup` (currently 5000). |
| `--alternative-set` | Optional | `fixreal` | CSV path for fixed alternatives. Uses the same CSV schema as fixreal batch CSV (`real_profile_id` + all attributes). Cannot be used with `--n-makeup`. |
| `--n-top` | Optional | `top` | Number of top-scored profiles to compare against. |
| `--output` | Optional | most modes | Output **filename** or **folder** (see output rules below). |
| `--api-key-env` | Optional | collector modes | Env var name holding the API key (defaults to config `openai.api_key_env_var`). |
| `--api-key-envs` | Optional | `basic`, `fixreal` | Comma-separated API key env vars for parallel runs. `basic`: splits `[start,end)`. `fixreal`: shards CSV batch rows by `real_profile_id`. |
| `--reasoning-effort` | Optional | collector modes | Overrides config `openai.reasoning_effort` (note: ignored when `--logprobs on`; see below). |
| `--logprobs` | Optional | collector modes | `on` or `off`; overrides config `openai.logprobs.enabled`. Adds `prob_chosen` / `prob_nochosen` columns. |
| `--sample-ids-file` | Optional | `context` | NPY filename under `data/` storing sampled makeup profile ids. Default: `sample5k_profile_ids.npy`. |
| `--scored-limit` | Optional | `context` | Limits the rows considered when sampling. |
| `--context-date` | Optional | `context` | Injected “current date” string used in the system prompt. |
| `--rag-faiss` | Optional* | `rag-faiss` | Path to FAISS index (or set env `RAG_FAISS`). |
| `--rag-meta` | Optional* | `rag-faiss` | Path to metadata jsonl (or set env `RAG_META`). |
| `--rag-k` | Optional | `rag-faiss` | Top-k chunks to retrieve (default: 3). |
| `--rag-per-chars` | Optional | `rag-faiss` | Max chars per retrieved chunk (default: 1200). |
| `--rag-embed-model` | Optional | `rag-faiss` | Embedding model name for retrieval. |
| `--exclude-ids-file` | Optional | `rag-faiss` | NPY filename under `data/` listing profile_ids to exclude from sampling. |
| `--model` | Optional | `rag` | Override model name (rag only; forwarded to `RAG_langchain`). |
| `--temperature` | Optional | `rag` | Override temperature (rag only; forwarded to `RAG_langchain`). |

Notes:
- `Yes*` means “required only for that experiment mode”.
- For `rag-faiss`, `--rag-faiss`/`--rag-meta` are required via flags or env vars.
- For `fixreal`, `--api-key-envs` requires `--real-profile` to be a CSV path (batch mode).

### Fixreal: `--real-profile` can be a CSV path (batch mode)

For `--experiment fixreal`, you may set `--real-profile` to a `.csv` path. The collector will run fixreal once per row.

CSV requirements:
- Must contain a `real_profile_id` column.
- Must contain **all attributes** as columns, using either:
  - config keys (e.g., `battery_life`, `screen_size`, ...), or
  - config display names (e.g., `battery life (in hours of video playback)`, ...).

Output behavior:
- Writes one file per row as `{real_profile_id_with_underscores}_fixreal{n_makeup}.csv`.
- If `--output` is provided in batch mode, it must be a **folder** (not a filename).

### Fixreal: `--alternative-set` uses a fixed alternatives CSV

For `--experiment fixreal`, you may set `--alternative-set` to a CSV file. The collector compares each real profile against rows from this CSV instead of sampling from generated makeup profiles.

CSV requirements:
- Same schema as fixreal batch CSV for `--real-profile`:
  - Must contain `real_profile_id` (used as alternative profile id)
  - Must contain all attributes as columns (keys or display names)

Rules:
- `--alternative-set` is only supported in `fixreal`.
- `--alternative-set` and `--n-makeup` are mutually exclusive.

Default output naming:
- Single real profile: `{real_profile_id_with_underscores}_fixreal_altset{num_alternatives}.csv`
- Batch real profiles (`--real-profile` is CSV): one file per row with the same suffix.

### Output rules (`--output`)

If `--output` is omitted, outputs go under `output/` with default filenames.

If `--output` **looks like a folder path** (e.g. ends with `/`, exists as a directory, or has no extension), outputs are written under that folder:
- Relative paths are interpreted under the project root (e.g., `--output experiments/experiment_1/` → `experiments/experiment_1/`).
- Absolute paths are used as-is.

If `--output` looks like a filename, it is used as the exact output file name (single-output modes only).

### Sampling reuse (`n_makeup`)

Fixreal sampling reuses a deterministic cached id list:
- When `n_makeup=5000`, the collector writes/reads `data/sample5000_profile_ids.npy`.
- If the file exists, it is reused (no re-sampling), keeping comparisons consistent across runs.

### Resuming interrupted fixreal runs

Fixreal supports resuming from a partial output CSV:
- If the output file already exists (e.g., `{real_profile_id}_fixreal5000.csv`), the collector reads existing rows, detects completed `pair_id` values, and **skips them**.
- Re-run the same command to continue until all `pair_id` values are completed.

### Logprobs vs reasoning effort

When `--logprobs on`, the collector uses a separate “logprobs model” config and does **not** pass `reasoning_effort` to the API call (so `--reasoning-effort` is effectively ignored for those calls).

### Examples

Basic:
```bash
llm-collect --experiment basic --start 0 --end 10000
```

Basic (parallel):
```bash
llm-collect --experiment basic --start 0 --end 10000 \
  --api-key-envs OPENAI_KEY_1,OPENAI_KEY_2,OPENAI_KEY_3
```

Fixreal (single id from config):
```bash
llm-collect --experiment fixreal --real-profile "iPhone 16 Pro" --n-makeup 5000
```

Fixreal (CSV batch, write into a folder):
```bash
llm-collect --experiment fixreal --real-profile data/custom_real_profiles.csv \
  --n-makeup 5000 --output batch_outputs/
```

Fixreal batch (parallel by real-profile with multiple API keys):
```bash
llm-collect --experiment fixreal --real-profile data/custom_real_profiles.csv \
  --n-makeup 5000 --api-key-envs OPENAI_KEY_1,OPENAI_KEY_2,OPENAI_KEY_3 \
  --output batch_outputs/
```

Fixreal (fixed alternatives from CSV):
```bash
llm-collect --experiment fixreal --real-profile "iPhone 16 Pro" \
  --alternative-set experiments/alternatives/design100.csv
```

Top:
```bash
llm-collect --experiment top --real-profile "iPhone 16 Pro" --n-top 50
```

Context:
```bash
llm-collect --experiment context --real-profile "iPhone 16 Pro" --context data/re16.txt
```

RAG (LangChain pipeline in `RAG_langchain/`):
```bash
llm-collect --experiment rag --real-profile "iPhone 16 Pro" --api-key-env OPENAI_API_KEY
```

RAG (FAISS):
```bash
llm-collect --experiment rag-faiss --real-profile "iPhone 16 Pro" \
  --rag-faiss path/to/index.faiss --rag-meta path/to/records.jsonl
```

## Models

| Model | Description | File |
|-------|-------------|------|
| LogisticRegression | Linear baseline model | `llm_belief/models/scoring.py` |
| MLPScorer | Multi-layer perceptron | `llm_belief/models/scoring.py` |
| MLPAttentionScore | MLP with attention | `llm_belief/models/scoring.py` |
| LinearInteractionModel | Adaptive Lasso with interactions | `llm_belief/models/adalasso.py` |
| XGBoost | Gradient boosting for ranking | `llm_belief/models/xgboost_model.py` |


## Analysis Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/dimensionality_reduction/probability_all_models_FA.ipynb` | FA + Varimax dimensionality reduction |
| `notebooks/dimensionality_reduction/probability_all_models_PCA.ipynb` | PCA dimensionality reduction |
| `notebooks/dimensionality_reduction/probability_all_models_kernel_PCA.ipynb` | Kernel PCA |
| `notebooks/interpretability/integrated_gradient.ipynb` | Feature attribution analysis |
| `notebooks/interpretability/partial_dependent_plot.ipynb` | Partial dependence plots |

Note: notebooks are for exploratory analysis; production code lives in `llm_belief/` and `scripts/`.

## iPhone Attributes

The following attributes are used for iPhone profile generation:

| Attribute | Values |
|-----------|--------|
| Battery Life | 18, 24, 30, 36, 42 hours |
| Screen Size | 6.1, 6.3, 6.6, 6.9 inches |
| Thickness | 6, 7.7, 8.3, 8.8 mm |
| Front Camera | 12, 18, 24, 30 MP |
| Rear Camera | 36, 48, 60, 72 MP |
| Focal Length | 1, 3, 5, 8, 10x |
| Ultrawide Camera | equipped / not equipped |
| Geekbench Score | 6200, 7000, 7800, 8600, 9400 |
| RAM | 4, 8, 12, 16 GB |
| Price | $749, $849, $949, $1049, $1149, $1249 |

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black llm_belief/ scripts/
isort llm_belief/ scripts/
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm_belief_elicitation,
  title={},
  author={},
  year={},
  url={}
}
```
