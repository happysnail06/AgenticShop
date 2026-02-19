"""
GPT Search Product Curation:
    Run GPT web search to curate products for each user based on prepared search inputs.

    Usage example:
        python src/search_llms_scripts/gpt_search.py --category clothing --samples 1
        python src/search_llms_scripts/gpt_search.py --category electronics --samples 5
        python src/search_llms_scripts/gpt_search.py --input_path product_curation_artifacts/inputs/clothing_search_input.json --samples 3
"""

from pydantic import BaseModel, Field
import argparse
import asyncio
from tqdm import tqdm
from openai import AsyncOpenAI
import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv


class SearchResult(BaseModel):
    product_name: str = Field(description="The name of the product")
    product_url: str = Field(description="The URL of the product page")
    reasoning: str = Field(description="Why the product fits the user's context and needs")


class SearchResultsPayload(BaseModel):
    search_results: List[SearchResult] = Field(description="Exactly three products with URL and reasoning")


SYSTEM_PROMPT = "You are an AI assistant that helps with product curation. Make sure you provide the correct URL for the product. The URL should be a direct link to the product page."

CATEGORY_INPUT_MAPPING = {
    "clothing": "clothing_search_input.json",
    "electronics": "electronics_search_input.json",
    "grocery_attribute_specific": "grocery_attribute_specific_search_input.json",
    "grocery_brand_categorical": "grocery_brand_categorical_search_input.json",
    "grocery_explicit_title": "grocery_explicit_title_search_input.json",
    "grocery_explicit": "grocery_explicit_title_search_input.json",
    "home": "home_search_input.json",
    "open": "open_search_input.json",
}


def get_project_paths():
    """Get project root and other important paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return {
        "project_root": project_root,
        "env_path": os.path.join(project_root, ".env.local"),
        "input_dir": os.path.join(project_root, "product_curation_artifacts", "inputs"),
        "output_dir": os.path.join(project_root, "product_curation_artifacts", "search_llms", "gpt"),
    }


def load_existing_results(output_path: str) -> Dict[str, Any]:
    """Load existing results for incremental saving (skip already completed users)."""
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {entry["user"]: entry for entry in existing if entry.get("search_results")}
    return {}


def save_results(output_path: str, results: List[Dict[str, Any]]):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


async def get_search_result(client: AsyncOpenAI, prompt: str) -> Dict[str, Any]:
    """Call OpenAI API with web search to get product recommendations."""
    response = await client.responses.parse(
        model="gpt-4o",
        input=SYSTEM_PROMPT + prompt,
        text_format=SearchResultsPayload,
        tools=[{"type": "web_search"}],
        include=["web_search_call.action.sources"],
    )
    parsed: SearchResultsPayload = response.output_parsed
    return parsed.model_dump()


async def process_sample(client: AsyncOpenAI, sample: Dict[str, Any], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Process a single sample with concurrency control and error handling."""
    user_id = sample["user"]
    prompt = sample["prompt"]

    async with semaphore:
        try:
            response = await get_search_result(client, prompt)
            return {
                "user": user_id,
                "search_results": response.get("search_results", []),
            }
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            return {
                "user": user_id,
                "search_results": [],
                "error": str(e),
            }


def argparser():
    parser = argparse.ArgumentParser(description="GPT search product curation")
    parser.add_argument("--category", type=str, help="Category key, e.g., clothing, electronics, home, open")
    parser.add_argument("--input_path", type=str, help="Direct path to search input JSON (overrides --category)")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to process")
    return parser.parse_args()


async def main():
    paths = get_project_paths()
    load_dotenv(paths["env_path"])

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env.local")

    client = AsyncOpenAI(api_key=api_key)
    args = argparser()

    # Resolve input path
    if args.input_path:
        input_path = os.path.abspath(args.input_path)
    elif args.category:
        filename = CATEGORY_INPUT_MAPPING.get(args.category)
        if not filename:
            raise ValueError(f"Unknown category '{args.category}'. Valid: {list(CATEGORY_INPUT_MAPPING.keys())}")
        input_path = os.path.join(paths["input_dir"], filename)
    else:
        raise ValueError("Provide either --category or --input_path")

    # Load search inputs
    with open(input_path, "r", encoding="utf-8") as f:
        search_inputs: List[Dict[str, Any]] = json.load(f)
    samples = search_inputs[: args.samples]

    # Setup output
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}_output.json")

    # Load existing results to skip completed users
    completed = load_existing_results(output_path)
    pending = [s for s in samples if s["user"] not in completed]
    print(f"Total: {len(samples)}, Already completed: {len(completed)}, Pending: {len(pending)}")

    if not pending:
        print("All samples already completed.")
        return

    # Collect all results (existing + new)
    all_results = list(completed.values())

    # Process with concurrency limit
    semaphore = asyncio.Semaphore(10)
    tasks = [process_sample(client, sample, semaphore) for sample in pending]

    with tqdm(total=len(tasks), desc="Searching products") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            all_results.append(result)
            save_results(output_path, all_results)
            pbar.update(1)
            if result.get("user"):
                pbar.set_postfix({"user": result["user"][:16]})

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
