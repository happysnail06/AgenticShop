# AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping

This is code repository for AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping

📄 **Paper Link**: [**AgenticShop**](https://drive.google.com/file/d/1ZrK7A7az16I9bTVAnrXueytwcTggH48g/view?usp=sharing)

🎉 **Our paper has been Accepted at [The Web Conference 2026](https://www2026.thewebconf.org/index.html)**

## Introduction

AgenticShop is a benchmark for evaluating how well agentic systems curate personalized products in open-web shopping environments. It captures realistic shopping intents, diverse user profiles, and fine-grained preferences, and introduces a checklist-driven evaluation framework grounded in verifiable product evidence to measure true personalization beyond simple product search.

---

![System Architecture](assets/overview.png)


## Setup

### 1. Create Python Virtual Environment
```bash
conda create -n agenticshop python=3.10.13
conda activate agenticshop
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Environment Configuration
Copy the environment template and add your API keys:
```bash
cp env.example .env.local
# Edit .env.local and add your OpenAI API key
```

## Data

Raw purchase history data is provided in `data/user_raw/` for benchmark construction:

| Domain | Users |
|--------|-------|
| Clothing | 104 |
| Electronics | 83 |
| Home | 80 |
| Grocery | 50 |
| Open | 50 |

Our dataset is built upon the well-established [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset and can be expanded to additional users and domains it supports.

Sample user profile data is available in `data/user_profiles_samples/` to help you get started with the benchmark.

## Project Structure
- `src/benchmark_construction/` - Scripts for generating benchmark data
- `src/search_llms_scripts/` - Example product curation scripts (GPT search provided as reference)
- `product_curation_artifacts/` - Inputs and outputs for product curation
- `src/benchmark_evaluation/` - Evaluation framework and modules
- `eval_results/` - Evaluation outputs and results

## Pipeline

The workflow consists of three phases. The generated user profiles serve as inputs to agentic systems, which curate product recommendations. The resulting product links are then evaluated against each user's checklist.

### Phase 1 — Benchmark Construction

Generate user contexts, queries, and evaluation checklists from raw purchase histories.

| | Path | Description |
|---|---|---|
| **Input** | `data/user_raw/` | Raw purchase histories |
| **Output** | `data/user_profiles/{domain}_user_profile.json` | User contexts, queries, and evaluation checklists |

```bash
python src/benchmark_construction/1_gen_user_context.py --domain clothing --samples 1
python src/benchmark_construction/2_gen_user_query.py --domain clothing --samples 1
python src/benchmark_construction/3_gen_user_checklist.py --domain clothing --samples 1
python src/benchmark_construction/4_organize_user_profile.py --domain clothing
```

### Phase 2 — Product Curation

Prepare search inputs from user profiles and run agentic systems to curate products. We provide a GPT search script as an example — any agentic system can be used in its place.

| | Path | Description |
|---|---|---|
| **Input** | `data/user_profiles/{domain}_user_profile.json` | User profiles from Phase 1 |
| **Output** | `product_curation_artifacts/inputs/{category}_search_input.json` | Search prompts and evaluation checklists |
| **Output** | `product_curation_artifacts/{model_type}/{model_name}/{category}_search_output.json` | Curated product URLs per user |

```bash
python src/search_llms_scripts/prepare_search_input.py data/user_profiles/clothing_user_profile.json
python src/search_llms_scripts/gpt_search.py --category clothing --samples 1
```

### Phase 3 — Benchmark Evaluation

Extract product information from curated URLs and score against user checklists across 6 dimensions.

| | Path | Description |
|---|---|---|
| **Input** | `product_curation_artifacts/` | User checklists and curated product URLs |
| **Output** | `eval_results/{...}/{category}/user_{id}/result_{N}/` | `N`-th curated product's extraction and evaluation results |

```bash
python src/benchmark_evaluation/run_pipeline.py \
  --model-type search_llms \
  --model-name gpt \
  --category clothing \
  --num-users 1
```
