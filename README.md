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
The workflow consists of three main phases:


**Phase 1 — Benchmark Construction**: Generate user contexts, queries, and evaluation checklists that form the foundation of the benchmark dataset.

- **Input**: `data/user_raw/` (raw purchase histories)
- **Output**: `data/user_profiles/{domain}_user_profile.json` (final user profiles with queries and checklists)

```bash
# Step 1: Generate diverse user contexts with shopping preferences and behaviors
python src/benchmark_construction/1_gen_user_context.py --domain clothing --samples 1

# Step 2: Create realistic user queries based on the generated contexts
python src/benchmark_construction/2_gen_user_query.py --domain clothing --samples 1

# Step 3: Build evaluation checklists tailored to each user's preferences
python src/benchmark_construction/3_gen_user_checklist.py --domain clothing --samples 1

# Step 4: Organize outputs into final user profiles
python src/benchmark_construction/4_organize_user_profile.py --domain clothing
```

The generated user profiles — including shopping queries — serve as inputs to agentic systems, which curate product recommendations. The resulting product links are then evaluated against the user's checklist.

**Phase 2 — Product Curation**: Prepare search inputs from user profiles and run agentic systems to curate products. We provide a GPT search script as an example — any agentic system can be used in its place.

- **Input**: `data/user_profiles/{domain}_user_profile.json`
- **Output**:
  - `product_curation_artifacts/inputs/{category}_search_input.json` (search prompts and evaluation checklists)
  - `product_curation_artifacts/{model_type}/{model_name}/{category}_search_input_output.json` (curated product URLs)

```bash
# Step 1: Build search prompts from user profiles
python src/search_llms_scripts/prepare_search_input.py data/user_profiles/clothing_user_profile.json

# Step 2: Run search LLM to curate products (example using GPT)
python src/search_llms_scripts/gpt_search.py --category clothing --samples 1
```

**Phase 3 — Benchmark Evaluation**: Run the evaluation pipeline to extract product information and score against user checklists.

- **Input**: `product_curation_artifacts/` (user checklists and curated product URLs)
- **Output**: `eval_results/{model_type}/{model_name}/{category}/user_{id}/result_{N}/` containing:
  - `extraction_result.json` — extracted product attributes
  - `evaluation_result.json` — checklist satisfaction scores across 6 dimensions

```bash
# Run complete evaluation pipeline with example parameters:
# --model-type: Type of model (search_llms or web_agents)
# --model-name: Specific model name (gpt, claude, etc.)
# --category: Product category (clothing, electronics, home, etc.)
# --num-users: Number of users to evaluate
python src/benchmark_evaluation/run_pipeline.py \
  --model-type search_llms \
  --model-name gpt \
  --category clothing \
  --num-users 1
```
