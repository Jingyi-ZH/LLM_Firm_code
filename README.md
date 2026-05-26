# Can Large Language Models Predict Firms’ New Product Launches?

This repository contains the code and supporting artifacts for eliciting product-specification beliefs from large language models using pairwise conjoint-style comparisons.

The current package centers on iPhone specification experiments, with reusable code for generating product profiles, collecting LLM pairwise choices, training ranking/scoring models, and producing analysis notebooks and figures. Domain-specific cases are configured through YAML app specs under `config/apps/`.

## What Is Included

- `llm_belief/`: core Python package for profile generation, data collection, preprocessing, models, and visualization.
- `config/`: global configuration plus app-specific YAML files.
- `scripts/run_collection.py`: main CLI for LLM data collection.
- `scripts/run_preprocessing.py`: profile generation CLI.
- `scripts/run_training.py`: model training CLI.
- `data/`: input and derived data used by the analysis workflow.
- `output/`: collected LLM response outputs.
- `plot/`: generated figures.
- `notebooks/`: exploratory and analysis notebooks.
- `iPhone17/`: iPhone 17-specific profiles, RAG script, external-knowledge utilities, and analysis notebook.

Some local or legacy folders are intentionally ignored by Git, including `RAG_langchain/`, `active_learn/`, `experiments/`, several application scratch folders, local virtual environments, logs, and large generated artifacts.

## Setup

Use Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For notebook work or development tools:

```bash
pip install -e ".[dev]"
```

For FAISS/RAG utilities:

```bash
pip install -e ".[rag]"
```

## API Configuration

Copy the environment template and add your OpenAI API key only if you will rerun API collection steps:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

Most analysis and plotting workflows can use the committed/generated `data/`, `output/`, and `plot/` artifacts without rerunning paid API calls.

## Configuration

Global defaults live in `config/config.yaml`. App-specific prompt text, attributes, and real profiles live in `config/apps/`.

The active app spec is controlled by `config/config.yaml` under:

```yaml
app:
  spec_path: "config/apps/iphone.yaml"
```

You can override the app spec without editing files:

```bash
export LLM_BELIEF_APP_SPEC_PATH=config/apps/iphone17.yaml
```

Available app specs currently include:

- `config/apps/iphone.yaml`
- `config/apps/iphone17.yaml`
- `config/apps/iphone17e.yaml`
- `config/apps/iphone17e_basestorage.yaml`
- `config/apps/ipadmini7.yaml`
- `config/apps/playstation5pro.yaml`
- `config/apps/Nvidia5090.yaml`
- `config/apps/applewatch.yaml`

## Reproduction Workflow

The high-level workflow is:

1. Generate hypothetical product profiles.
2. Collect LLM pairwise choices.
3. Train scoring/ranking models.
4. Compare real or candidate profiles against model beliefs.
5. Use notebooks to regenerate tables and figures.

Generate profile data:

```bash
python scripts/run_preprocessing.py --task generate-profiles
```

Collect basic pairwise comparisons:

```bash
python scripts/run_collection.py --experiment basic --start 0 --end 1000
```

Run fixreal comparisons against sampled hypothetical profiles:

```bash
python scripts/run_collection.py --experiment fixreal \
  --real-profile "iPhone 16 Pro" \
  --n-makeup 5000
```

Run all pairwise combinations across a CSV of candidate/real profiles:

```bash
python scripts/run_collection.py --experiment allcomb \
  --real-profile iPhone17/represents104.csv \
  --output output/iPhone17/
```

Train a model from collected pairwise outputs:

```bash
python scripts/run_training.py --model logistic --input-glob "output/*_*.csv"
python scripts/run_training.py --model mlp --input-glob "output/*_*.csv"
python scripts/run_training.py --model xgboost --input-glob "output/*_*.csv"
```

Open analysis notebooks under `notebooks/` and domain-specific folders such as `iPhone17/` to reproduce tables and plots.

## Data Collection CLI

The main collection entry point is:

```bash
python scripts/run_collection.py --help
```

Installed entry point:

```bash
llm-collect --help
```

Supported experiment modes:

- `basic`: pairwise comparisons between generated hypothetical profiles.
- `fixreal`: one real profile versus sampled hypothetical profiles.
- `top`: one real profile versus top-scored hypothetical profiles.
- `context`: fixreal with injected text context.
- `rag-faiss`: fixreal with a FAISS index and metadata jsonl.
- `allcomb`: all pairwise combinations across a CSV of profiles.
- `question-csv`: one free-form question per CSV row.

Common examples:

```bash
llm-collect --experiment basic --start 0 --end 10000
```

```bash
llm-collect --experiment fixreal \
  --real-profile "iPhone 16 Pro" \
  --n-makeup 5000
```

```bash
llm-collect --experiment context \
  --real-profile "iPhone 16 Pro" \
  --context data/re16.txt
```

```bash
llm-collect --experiment rag-faiss \
  --real-profile "iPhone 16 Pro" \
  --rag-faiss path/to/index.faiss \
  --rag-meta path/to/records.jsonl
```

```bash
llm-collect --experiment question-csv \
  --question-csv data/questions_sanity_iphone.csv \
  --question-column question \
  --product "iPhone"
```

For the complete and current argument list, use `python scripts/run_collection.py --help`.

## iPhone 17 RAG Case

The iPhone 17 RAG workflow is implemented independently of the ignored legacy `RAG_langchain/` folder.

Main script:

```bash
python iPhone17/run_allcomb_rag_langchain.py
```

Default inputs:

- Candidate profiles: `iPhone17/represents104.csv`
- Static context: `data/re16.txt`
- RAG source HTML: `data/external_knowledge/macrumor_iPhone17.html`
- App spec: `config/apps/iphone17.yaml`

Default output:

```text
output/iPhone17/represents104_allcomb_RAG.csv
```

The script:

1. Loads 104 candidate profiles from `iPhone17/represents104.csv`.
2. Builds all pairwise combinations, giving 5,356 comparisons.
3. Cleans the MacRumors HTML source with `iPhone17/html_to_text.py`.
4. Splits the cleaned text into overlapping chunks.
5. Embeds chunks with OpenAI embeddings and stores them in an in-memory vector store.
6. Retrieves the top-k relevant chunks for each pairwise prompt.
7. Prepends static context and retrieved context as system messages.
8. Calls the shared `PairwiseCollector` API logic.
9. Writes pairwise choices, retrieved context, retrieved hit metadata, and selected profile ids to CSV.

Useful overrides:

```bash
python iPhone17/run_allcomb_rag_langchain.py \
  --real-profile-csv iPhone17/represents104.csv \
  --html data/external_knowledge/macrumor_iPhone17.html \
  --context data/re16.txt \
  --output output/iPhone17/represents104_allcomb_RAG.csv \
  --rag-k 3 \
  --chunk-size 1000 \
  --chunk-overlap 200
```

The output is resumable: if the output CSV already exists, completed `pair_id` values are skipped.

## Output And Generated Files

Generated files are written mainly under:

- `data/`
- `output/`
- `plot/`
- `logs/`

Some generated caches are intentionally ignored by Git. For example, `data/sample5000_profile_ids.npy` is regenerated/reused locally by fixreal sampling and is not tracked.

`output/` can be large because it stores LLM response CSVs. Keep generated outputs only when they are required for replication.

## Models

Implemented model classes live under `llm_belief/models/`:

- `LogisticRegression`
- `MLPScorer`
- `MLPAttentionScore`
- `LinearInteractionModel`
- `train_xgb_pairwise`
- `XGBScorerTorch`

Training entry point:

```bash
python scripts/run_training.py --help
```

Some older notebooks may depend on local analysis modules or generated files that are not part of the cleaned replication package. Prefer the core package modules and documented scripts for rerunnable workflows.

## License

MIT License. See `LICENSE`.

## Citation

If you use this code in research, please cite the associated paper or repository:

```bibtex
@misc{llm_product_launch_pred,
  title = {Can Large Language Models Predict Firms’ New Product Launches?},
  author = {},
  year = {},
  url = {}
}
```
