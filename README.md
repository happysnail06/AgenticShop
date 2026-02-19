# AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping

This is code repository for AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping

📄 **Paper Link**: [**AgenticShop**](https://drive.google.com/file/d/1ZrK7A7az16I9bTVAnrXueytwcTggH48g/view?usp=sharing)

🎉 **Our paper has been Accepted at [The Web Conference 2026](https://www2026.thewebconf.org/index.html)**

## Introduction

AgenticShop is a benchmark for evaluating how well agentic systems curate personalized products in open-web shopping environments. It captures realistic shopping intents, diverse user profiles, and fine-grained preferences, and introduces a checklist-driven evaluation framework grounded in verifiable product evidence to measure true personalization beyond simple product search.

---

![System Architecture](assets/overview.png)


## Pipeline
The workflow consists of two main phases:


**Benchmark Construction**: Generate user contexts, queries, and evaluation checklists that form the foundation of the benchmark dataset.

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

**Benchmark Evaluation**: Run the evaluation pipeline to test different models and approaches against the constructed benchmark, measuring their performance in product curation tasks.

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

## Setup

### 1. Create Python Virtual Environment
```bash
conda create -n agenticshop python=3.10.13
conda activate agenticshop
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the environment template and add your API keys:
```bash
cp env.example .env.local
# Edit .env.local and add your OpenAI API key
```

### 4. Project Structure
- `src/benchmark_construction/` - Scripts for generating benchmark data
- `src/benchmark_evaluation/` - Evaluation framework and modules
- `eval_inputs/` - Input data for evaluation
- `eval_results/` - Evaluation outputs and results

## Data

Sample user profile data is available in `data/user_profiles_samples/` to help you get started with the benchmark. This includes:

- Pre-generated user contexts and queries
- Sample evaluation checklists
