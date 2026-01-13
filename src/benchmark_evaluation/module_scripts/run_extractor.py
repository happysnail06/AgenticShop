#!/usr/bin/env python3
"""
Main extraction script using refactored class-based extractor.

Purpose: 
    - Uses the Extractor class for product information extraction only
    - Reads new eval input format: { user_id: { "search_results": [ { product_url, ... } ] } }
    - Builds input/output paths from model_name and category arguments
    - Concurrency model: categories concurrent; users sequential; items within a user concurrent

Usage:
    python run_extractor.py \
      --model-type search_llms \
      --model-name gpt \
      --category clothing \
      --num-users 1
      [--all-users] [--all-categories]

Categories:
  clothing | electronics | home | open | grocery_attribute | grocery_brand | grocery_explicit
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Ensure the src directory is on sys.path for direct execution
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from benchmark_evaluation.modules import Extractor

# Category mapping for convenience
CATEGORY_MAPPING: Dict[str, str] = {
    'grocery': 'Grocery_and_Gourmet_Food',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'electronics': 'Electronics',
    'home': 'Home_and_Kitchen',
    'open': 'open_ended_curation',
}

def infer_category_and_variant(input_path: Path) -> tuple[str, str]:
    """Infer category name and variant from the input filename.

    Examples:
    - Clothing_Shoes_and_Jewelry_eval_input.json -> (Clothing_Shoes_and_Jewelry, default)
    - Grocery_and_Gourmet_Food_eval_input_explicit.json -> (Grocery_and_Gourmet_Food, explicit)
    - open_ended_curation_eval_input.json -> (open_ended_curation, default)
    """
    name = input_path.name
    if name.endswith('_eval_input.json'):
        category = name.replace('_eval_input.json', '')
        return category, 'default'
    # expect pattern *_eval_input_<variant>.json
    prefix, _, variant_part = name.partition('_eval_input_')
    variant = variant_part.replace('.json', '')
    return prefix, variant


async def process_user_items(user_id: str, user_items: List[Dict[str, Any]], 
                           extractor: Extractor, base_out: Path,
                           error_records: List[Dict[str, Any]] | None = None) -> None:
    """Process all items for a single user using asyncio.
    
    Args:
        user_id (str): User identifier.
        user_items (List[Dict[str, Any]]): List of items to process for this user.
        extractor (Extractor): Extractor instance.
        base_out (Path): Base output directory.
    """
    
    
    # print(f"Processing user {user_id} with {len(user_items)} items...")
    # Process all items for this user concurrently
    tasks: List[Any] = []
    urls: List[str] = []
    for i, item in enumerate(user_items):
        out_idx = item.get("original_index", i)
        
        # Create user-specific subdirectory, preserving original index when provided
        user_out = base_out / f"user_{user_id}" / f"result_{out_idx:03d}"
        user_out.mkdir(parents=True, exist_ok=True)
        
        # Create task for this item, passing the output index
        task = extractor.process_item(out_idx, item, user_out)
        tasks.append(task)
        urls.append(item.get("product_url", ""))
            
    
    # Execute all tasks concurrently for this user
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for any exceptions in the results
        if error_records is not None:
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_records.append({
                        "user_id": user_id,
                        "url": urls[i] if i < len(urls) else "",
                        "error": str(result),
                    })


def process_user_sync(user_id: str, user_items: List[Dict[str, Any]], 
                     client: OpenAI, base_out: Path,
                     error_records: List[Dict[str, Any]] | None = None) -> None:
    """Synchronous wrapper for processing user items.
    
    Args:
        user_id (str): User identifier.
        user_items (List[Dict[str, Any]]): List of items to process for this user.
        client (OpenAI): OpenAI client instance.
        base_out (Path): Base output directory.
    """
    # Create extractor instance for this thread
    extractor = Extractor(client)
    
    # Run the async function in a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(process_user_items(user_id, user_items, extractor, base_out, error_records))
    finally:
        loop.close()


def build_paths(model_type: str, model_name: str, category_key: str) -> tuple[Path, Path, str, str]:
    """Build input and output base paths, and return (input_path, output_dir, category_full, variant)."""
    input_base = Path(f"/work/agent-as-judge/data/evaluation/input_collections/{model_type}/{model_name}")
    output_base = Path(f"/work/AgenticShop/eval_results/{model_type}/{model_name}")

    grocery_files = {
        'grocery_attribute': 'Grocery_and_Gourmet_Food_eval_input_attribute_specific.json',
        'grocery_brand': 'Grocery_and_Gourmet_Food_eval_input_brand_categorical.json',
        'grocery_explicit': 'Grocery_and_Gourmet_Food_eval_input_explicit.json',
    }

    if category_key in ('clothing', 'electronics', 'home', 'open'):
        category_full = CATEGORY_MAPPING[category_key]
        variant = 'default'
        filename = f"{category_full}_eval_input.json"
    else:
        # grocery variants
        filename = grocery_files[category_key]
        category_full = CATEGORY_MAPPING['grocery']
        # derive variant from filename suffix
        if 'attribute_specific' in filename:
            variant = 'attribute_specific'
        elif 'brand_categorical' in filename:
            variant = 'brand_categorical'
        else:
            variant = 'explicit'

    input_path = input_base / filename
    output_dir = output_base / category_full
    if variant != 'default':
        output_dir = output_dir / variant
    return input_path, output_dir, category_full, variant


async def main() -> None:
    """Main function using refactored class-based extractor for extraction only."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, choices=['web_agents','search_llms'], help='Model type (web_agents or search_llms)')
    parser.add_argument('--model-name', type=str, required=True, help='Model name key (e.g., gpt)')
    parser.add_argument('--category', type=str,
                       choices=['clothing','electronics','home','open','grocery_attribute','grocery_brand','grocery_explicit'],
                       help='Category key or grocery variant')
    parser.add_argument('--num-users', type=int, default=1, help='Number of users to process from the input file')
    parser.add_argument('--all-users', action='store_true', help='Process all users in the input file')
    parser.add_argument('--all-categories', action='store_true', help='Process all supported categories for the model')
    args = parser.parse_args()

    # Load environment from root directory
    root_dir = Path(__file__).parent.parent.parent
    load_dotenv(root_dir / '.env.local')

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('OPENAI_API_KEY not set')
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Determine categories to process
    supported_categories = ['clothing','electronics','home','open','grocery_attribute','grocery_brand','grocery_explicit']
    if args.all_categories:
        categories_to_process = supported_categories
    else:
        categories_to_process = [args.category]

    def process_category(cat_key: str, pbar_users: tqdm | None = None) -> None:
        input_path, output_dir, _, _ = build_paths(args.model_type, args.model_name, cat_key)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as f:
            eval_input: Dict[str, Any] = json.load(f)

        all_user_ids = list(eval_input.keys())
        user_ids = all_user_ids if args.all_users else all_user_ids[: max(1, args.num_users)]

        user_items_map: Dict[str, List[Dict[str, Any]]] = {}
        for user_id in user_ids:
            sr = eval_input[user_id]["search_results"][:3]
            pending_items: List[Dict[str, Any]] = []
            for idx, r in enumerate(sr):
                item_out = output_dir / f"user_{user_id}" / f"result_{idx:03d}" / "extraction_result.json"
                if not item_out.exists():
                    pending_items.append({
                        "product_url": r["product_url"],
                        "original_index": idx,
                    })
            if pending_items:
                user_items_map[user_id] = pending_items

        base_out = output_dir
        # Collect errors during user processing
        error_records: List[Dict[str, Any]] = []
        # Process users sequentially; items within a user run concurrently via asyncio
        for user_id, user_items in user_items_map.items():
            process_user_sync(user_id, user_items, client, base_out, error_records)
            if pbar_users is not None:
                pbar_users.update(1)

        # Write error log at the category directory if there were errors
        if error_records:
            log_path = output_dir / "error_logs.txt"
            with open(log_path, "w", encoding="utf-8") as lf:
                for rec in error_records:
                    lf.write(f"user_id={rec.get('user_id','')}\turl={rec.get('url','')}\terror={rec.get('error','')}\n")

    # Progress over categories
    with tqdm(total=len(categories_to_process), desc="Categories", unit="cat") as pbar_cats:
        # Run categories in parallel
        with ThreadPoolExecutor(max_workers=len(categories_to_process) or 1) as cat_pool:
            # one per-category user progress bar
            cat_to_user_pbar: Dict[str, tqdm] = {}
            def submit_category(cat_key: str):
                # lazy initialize user pbar size after loading file
                # create temp to compute length
                input_path, _, _, _ = build_paths(args.model_type, args.model_name, cat_key)
                with open(input_path, 'r', encoding='utf-8') as f:
                    eval_input: Dict[str, Any] = json.load(f)
                all_user_ids = list(eval_input.keys())
                user_ids = all_user_ids if args.all_users else all_user_ids[: max(1, args.num_users)]
                # compute pending count based on existing outputs
                input_path_check, output_dir_check, _, _ = build_paths(args.model_type, args.model_name, cat_key)
                _ = input_path_check  # unused
                pending = 0
                for user_id in user_ids:
                    sr = eval_input[user_id]["search_results"][:3]
                    for idx, _r in enumerate(sr):
                        item_out = output_dir_check / f"user_{user_id}" / f"result_{idx:03d}" / "extraction_result.json"
                        if not item_out.exists():
                            pending += 1
                cat_to_user_pbar[cat_key] = tqdm(total=pending, desc=f"{cat_key} users", unit="usr")
                return cat_pool.submit(process_category, cat_key, cat_to_user_pbar[cat_key])

            cat_futures = {submit_category(cat_key): cat_key for cat_key in categories_to_process}
            for fut in as_completed(cat_futures):
                cat_key = cat_futures[fut]
                fut.result()
                # close per-category pbar
                if cat_key in cat_to_user_pbar:
                    cat_to_user_pbar[cat_key].close()
                pbar_cats.update(1)

    print("All requested categories and users processed!")


if __name__ == '__main__':
    asyncio.run(main())
