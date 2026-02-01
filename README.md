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

```bash
# Basic pairwise comparison (10,000 pairs)
python scripts/run_collection.py --experiment basic --start 0 --end 1000
# Output: output/{start}_{end}.csv (e.g., output/0_1000.csv)

# Real vs. makeup profile comparison
python scripts/run_collection.py --experiment fixreal --real-profile "iPhone 16 Pro"
# Output: output/{real_profile_id_with_underscores}_fixreal{n_makeup}.csv

# Real vs. top scored profiles
python scripts/run_collection.py --experiment top --real-profile "iPhone 16 Pro" --n-top 50c
# Output: output/{real_profile_id_with_underscores}_ntop{n_top}.csv

# Real vs. makeup with injected context
python scripts/run_collection.py --experiment context --real-profile "iPhone 16 Pro" \
  --context data/re16.txt
# Output: output/context_{real_profile_id_with_underscores}_fixreal_{N}.csv
# Sampling: uses the same random sample as fixreal from `data/profiles_shuffled.csv`
# and saves/reads `data/sample{n_makeup}_profile_ids.npy`.

# RAG (default: RAG_langchain pipeline)
python scripts/run_collection.py --experiment rag --real-profile "iPhone 16 Pro" \
  --api-key-env OPENAI_API_KEY
# Output: output/rag_langchain_{real_profile_id}_fixreal{n_makeup}.csv
# Requires `output/{real_profile_id_with_underscores}_fixreal{n_makeup}.csv` from fixreal.

# Optional: custom FAISS-based RAG (requires index paths)
python scripts/run_collection.py --experiment rag-faiss --real-profile "iPhone 16 Pro" \
  --rag-faiss path/to/index.faiss --rag-meta path/to/records.jsonl
# Output: output/RAG_{real_profile_id_with_underscores}_fixreal_{n_makeup}.csv
```

Notes:
- Default RAG workflow lives in `RAG_langchain/` and is invoked via `--experiment rag`.
- There is also a separate custom RAG pipeline under `RAG/` (not default). If you have
  that folder, `bash scripts/run_rag_pipeline.sh` builds the FAISS index using
  `RAG/rag_build.py`.

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

### 1. Basic Pairwise Comparison
Compares 10,000 pairs of hypothetical iPhone profiles to learn LLM preferences.

```bash
python scripts/run_collection.py --experiment basic --start 0 --end 10000
```
LLM responses are recorded in `output/{start}_{end}.csv` (e.g., `output/0_10000.csv`).

### Parallel Basic Collection (Multiple API Keys)
Provide a comma-separated list of API key env vars to parallelize the range:

```bash
python scripts/run_collection.py --experiment basic --start 0 --end 10000 \
  --api-key-envs OPENAI_KEY_1,OPENAI_KEY_2,OPENAI_KEY_3,OPENAI_KEY_4,OPENAI_KEY_5
```

This splits `[start, end)` into N equal chunks (one per key) and runs them concurrently.

### Logprobs (Optional)
Enable logprobs to capture per-label probabilities in the CSV output.
This adds two columns: `prob_chosen` and `prob_nochosen`.
Supported for basic/fixreal/top/context/rag-faiss and rag (LangChain).

```bash
python scripts/run_collection.py --experiment basic --start 0 --end 10000 \
  --logprobs on
```

### 2. Real vs. Makeup Comparison
Compares real iPhone 16/17 specifications against makeup profiles.

```bash
python scripts/run_collection.py --experiment fixreal \
    --real-profile "iPhone 16 Pro" \
    --n-makeup 5000
```
LLM responses are recorded in `output/{real_profile_id_with_underscores}_fixreal{n_makeup}.csv`.
Sampled makeup profile ids are saved to `data/sample{n_makeup}_profile_ids.npy`.

### 3. Real vs. Top-Scored Comparison
Compares real specifications against top-scored profiles.

```bash
python scripts/run_collection.py --experiment top \
    --real-profile "iPhone 16 Pro" \
    --n-top 50
```
LLM responses are recorded in `output/{real_profile_id_with_underscores}_ntop{n_top}.csv`.
Requires `data/scored_profiles_shuffled.csv` (generated by training/scoring).
Generate it first via:
```bash
python scripts/run_training.py --model logistic --input-glob "output/*_*.csv"
```

### 4. Real vs. Makeup with Context Injection
Injects external context from a text file as system context.

```bash
python scripts/run_collection.py --experiment context \
    --real-profile "iPhone 16 Pro" \
    --context data/re16.txt
```
LLM responses are recorded in `output/context_{real_profile_id_with_underscores}_fixreal_{N}.csv`.
Sampling matches fixreal: random sample from `data/profiles_shuffled.csv`,
saved to `data/sample{n_makeup}_profile_ids.npy` (seed 2025).

Context examples:
- `data/re16.txt`: released iPhone 16 lineup specs (generated from config)
- `data/re16ru17.txt`: released iPhone specs + iPhone 17 rumor summary

### 5. RAG-Augmented Real vs. Makeup
Default RAG uses the LangChain pipeline in `RAG_langchain/`.
The notebook lives at `RAG_langchain/rag_langchain.ipynb`.

```bash
python scripts/run_collection.py --experiment rag \
    --real-profile "iPhone 16 Pro" \
    --api-key-env OPENAI_API_KEY
```
LLM responses are recorded in `output/rag_langchain_{real_profile_id}_fixreal{n_makeup}.csv`.
Requires `output/{real_profile_id_with_underscores}_fixreal{n_makeup}.csv` from fixreal.
Sampling matches fixreal: random sample from `data/profiles_shuffled.csv`,
saved to `data/sample{n_makeup}_profile_ids.npy` (seed 2025).

Optional: custom FAISS-based RAG (requires index paths):

```bash
python scripts/run_collection.py --experiment rag-faiss \
    --real-profile "iPhone 16 Pro" \
    --rag-faiss path/to/index.faiss \
    --rag-meta path/to/records.jsonl
```
LLM responses are recorded in `output/RAG_{real_profile_id_with_underscores}_fixreal_{n_makeup}.csv`.
Sampling matches fixreal: random sample from `data/profiles_shuffled.csv`,
saved to `data/sample{n_makeup}_profile_ids.npy` (seed 2025).

## `run_collection.py` CLI

### Purpose
Runs data-collection experiments for pairwise comparisons.

### Arguments
Required:
- `--experiment` (`basic|fixreal|top|context|rag|rag-faiss`)

Conditional required:
- `--start`, `--end` (required for `basic`)
- `--real-profile` (required for `fixreal`, `top`, `context`, `rag`, `rag-faiss`)
- `--context` (required for `context`)

Optional:
- `--n-makeup` (fixreal, rag, rag-faiss; default from `config/config.yaml` → `collection.default_n_makeup`)
- `--n-top` (top only, no default)
- `--output` (custom output filename)
- `--api-key-env` (single API key env var; optional)
- `--api-key-envs` (comma-separated API key env vars for parallel `basic`; optional)
- `--reasoning-effort` (override `openai.reasoning_effort`)
- `--logprobs` (`on|off`; overrides `openai.logprobs.enabled`)
- `--model` (rag only; override `openai.model`)
- `--temperature` (rag only; override `openai.temperature`)
- `--sample-ids-file` (context only)
- `--scored-limit` (context only)
- `--context-date` (context only)
- `--rag-faiss`, `--rag-meta` (rag-faiss only; or set `RAG_FAISS`/`RAG_META`)
- `--rag-k`, `--rag-per-chars`, `--rag-embed-model` (rag-faiss only)
- `--exclude-ids-file` (rag-faiss only)

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
