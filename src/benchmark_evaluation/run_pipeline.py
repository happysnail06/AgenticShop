#!/usr/bin/env python3
"""
Unified pipeline: Extraction → Evaluation

Purpose:
    - Runs extraction for selected categories/users
    - Then runs evaluation for those outputs
    - Parallelizes across categories; processes users sequentially; items per user run concurrently

Usage:
    python src/benchmark_evaluation/run_pipeline.py \
      --model-type search_llms \
      --model-name gpt \
      --category clothing \
      --num-users 1 \
      [--all-users] [--all-categories]
"""

import argparse
import asyncio
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from benchmark_evaluation.modules import Extractor, Evaluator


# Maps category keys to output directory names
CATEGORY_MAPPING: Dict[str, str] = {
    'grocery': 'Grocery_and_Gourmet_Food',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'electronics': 'Electronics',
    'home': 'Home_and_Kitchen',
    'open': 'open_ended_curation',
}

# Maps pipeline category keys to search input filenames in product_curation_artifacts/inputs/
SEARCH_INPUT_MAPPING: Dict[str, str] = {
    'clothing': 'clothing_search_input.json',
    'electronics': 'electronics_search_input.json',
    'home': 'home_search_input.json',
    'open': 'open_search_input.json',
    'grocery_attribute': 'grocery_attribute_specific_search_input.json',
    'grocery_brand': 'grocery_brand_categorical_search_input.json',
    'grocery_explicit': 'grocery_explicit_title_search_input.json',
}


def get_project_paths():
    """Get project root and other important paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return {
        "project_root": project_root,
        "env_path": os.path.join(project_root, ".env.local"),
    }


def build_paths(project_root: str, model_type: str, model_name: str, category_key: str) -> Tuple[Path, Path, Path, str]:
    """Build input/output/checklist paths for a given category.

    Args:
        project_root (str): Absolute path to the project root.
        model_type (str): Model family, e.g., 'web_agents' or 'search_llms'.
        model_name (str): Specific model identifier used in path layout.
        category_key (str): Short category key (e.g., 'clothing', 'grocery_attribute').

    Returns:
        Tuple of (search_output_path, eval_output_dir, checklist_input_path, variant).
    """
    root = Path(project_root)
    search_input_file = SEARCH_INPUT_MAPPING[category_key]
    search_output_file = search_input_file.replace('_search_input.json', '_search_output.json')

    # Search results from product curation
    search_output_path = root / "product_curation_artifacts" / model_type / model_name / search_output_file

    # Checklist source (search input file carries check_list from user profiles)
    checklist_input_path = root / "product_curation_artifacts" / "inputs" / search_input_file

    # Eval output directory
    if category_key.startswith('grocery_'):
        category_full = CATEGORY_MAPPING['grocery']
    else:
        category_full = CATEGORY_MAPPING[category_key]
    eval_output_dir = root / "eval_results" / model_type / model_name / category_full

    variant = 'default'
    if category_key == 'grocery_attribute':
        variant = 'attribute_specific'
    elif category_key == 'grocery_brand':
        variant = 'brand_categorical'
    elif category_key == 'grocery_explicit':
        variant = 'explicit'
    if variant != 'default':
        eval_output_dir = eval_output_dir / variant

    return search_output_path, eval_output_dir, checklist_input_path, variant


def load_search_results(path: Path) -> Dict[str, Any]:
    """Load search results from list format and convert to dict keyed by user ID.

    The search output is a JSON array: [{"user": "id", "search_results": [...]}, ...]
    This converts it to: {"id": {"search_results": [...]}, ...}
    """
    with open(path, 'r', encoding='utf-8') as f:
        results: List[Dict[str, Any]] = json.load(f)
    return {entry["user"]: entry for entry in results}


def load_checklists(path: Path) -> Dict[str, Any]:
    """Load checklists from search input file, keyed by user ID.

    The search input is a JSON array: [{"user": "id", "check_list": {...}}, ...]
    This converts it to: {"id": {"check_list": {...}}, ...}
    """
    with open(path, 'r', encoding='utf-8') as f:
        inputs: List[Dict[str, Any]] = json.load(f)
    return {entry["user"]: {"check_list": entry["check_list"]} for entry in inputs}


def discover_extraction_files(base_dir: Path, user_ids: List[str]) -> List[Tuple[str, List[Path]]]:
    """Find existing extraction result files for the specified users.

    Args:
        base_dir (Path): Category/variant output directory to search under.
        user_ids (List[str]): User IDs to include.

    Returns:
        List[Tuple[str, List[Path]]]: List of (user_id, sorted extraction files) pairs.
    """
    user_extractions_map: Dict[str, List[Path]] = {}
    pattern = str(base_dir / "user_*" / "result_*" / "extraction_result.json")
    extraction_files = glob.glob(pattern)
    for file_path in extraction_files:
        path_obj = Path(file_path)
        user_dir = path_obj.parent.parent.name
        user_id = user_dir[5:]  # strip "user_" prefix
        if user_id in user_ids:
            if user_id not in user_extractions_map:
                user_extractions_map[user_id] = []
            user_extractions_map[user_id].append(path_obj)
    for uid in user_extractions_map:
        user_extractions_map[uid].sort()
    return list(user_extractions_map.items())


async def extract_user_items(user_id: str, items: List[Dict[str, Any]], extractor: Extractor, base_out: Path,
                             error_records: List[Dict[str, Any]] | None = None) -> None:
    """Run extraction concurrently for a user's pending items."""
    tasks = []
    urls: List[str] = []
    for idx, item in enumerate(items):
        out_idx = item.get("original_index", idx)
        user_out = base_out / f"user_{user_id}" / f"result_{out_idx:03d}"
        user_out.mkdir(parents=True, exist_ok=True)
        tasks.append(extractor.process_item(out_idx, item, user_out))
        urls.append(item.get("product_url", ""))
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if error_records is not None:
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    error_records.append({
                        "user_id": user_id,
                        "url": urls[i] if i < len(urls) else "",
                        "error": str(res),
                    })


def extract_user_sync(user_id: str, items: List[Dict[str, Any]], client: OpenAI, base_out: Path,
                      error_records: List[Dict[str, Any]] | None = None) -> None:
    """Synchronous wrapper to run the async extraction for a user."""
    extractor = Extractor(client)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(extract_user_items(user_id, items, extractor, base_out, error_records))
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        try:
            shutdown_default_executor = getattr(loop, "shutdown_default_executor", None)
            if callable(shutdown_default_executor):
                loop.run_until_complete(shutdown_default_executor())
        except Exception:
            pass
        loop.close()


def evaluate_user(user_id: str, extraction_files: List[Path], checklists: Dict[str, Any], client: OpenAI) -> None:
    """Evaluate a user's extracted results against the checklist.

    Args:
        user_id (str): Target user identifier.
        extraction_files (List[Path]): Paths to prior extraction_result.json files.
        checklists (Dict[str, Any]): Loaded per-user checklists keyed by user ID.
        client (OpenAI): OpenAI client used by the evaluator.
    """
    evaluator = Evaluator(client)
    tasks: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
    check_list = checklists[user_id]["check_list"]
    for extraction_file in extraction_files:
        with open(extraction_file, 'r', encoding='utf-8') as f:
            model_response = json.load(f)
        model_data = {'search_output': model_response}
        tasks.append((extraction_file, model_data, check_list))
    reports = evaluator.evaluate_batch(tasks, max_workers=len(tasks))
    for i, (extraction_file, _, __) in enumerate(tasks):
        evaluation_file = extraction_file.parent / 'evaluation_result.json'
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(reports[i], f, ensure_ascii=False, indent=2)


def main() -> None:
    """Entry point: orchestrate extraction then evaluation for selected scopes."""
    parser = argparse.ArgumentParser(description="Unified extraction + evaluation pipeline")
    parser.add_argument('--model-type', type=str, required=True, choices=['web_agents','search_llms'])
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--category', type=str,
                        choices=['clothing','electronics','home','open','grocery_attribute','grocery_brand','grocery_explicit'])
    parser.add_argument('--num-users', type=int, default=1)
    parser.add_argument('--all-users', action='store_true')
    parser.add_argument('--all-categories', action='store_true')
    args = parser.parse_args()

    # Load environment from root directory
    paths = get_project_paths()
    load_dotenv(paths["env_path"])

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('OPENAI_API_KEY not set')
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    supported = ['clothing','electronics','home','open','grocery_attribute','grocery_brand','grocery_explicit']
    categories = supported if args.all_categories else [args.category]

    project_root = paths["project_root"]

    # Category progress
    with tqdm(total=len(categories), desc="Categories", unit="cat") as pbar_cats:
        def process_category(cat_key: str, pos: int) -> None:
            search_output_path, output_dir, checklist_input_path, _ = build_paths(
                project_root, args.model_type, args.model_name, cat_key
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Load search results (list format → dict keyed by user ID)
            eval_input = load_search_results(search_output_path)
            all_user_ids = list(eval_input.keys())
            user_ids = all_user_ids if args.all_users else all_user_ids[: max(1, args.num_users)]

            # Compute pending extraction items and build work map
            user_items_map: Dict[str, List[Dict[str, Any]]] = {}
            for user_id in user_ids:
                sr = eval_input[user_id]["search_results"][:3]
                pending: List[Dict[str, Any]] = []
                for idx, r in enumerate(sr):
                    item_out = output_dir / f"user_{user_id}" / f"result_{idx:03d}" / "extraction_result.json"
                    if not item_out.exists():
                        pending.append({
                            "product_url": r["product_url"],
                            "original_index": idx,
                        })
                if pending:
                    user_items_map[user_id] = pending

            remaining_users_extract = list(user_items_map.keys())
            print(f"[{cat_key}] Remaining users to extract: {len(remaining_users_extract)}" + (f" -> {', '.join(remaining_users_extract)}" if remaining_users_extract else ""))

            # Extract sequentially per user with items concurrent within user
            error_records: List[Dict[str, Any]] = []
            with tqdm(total=len(user_items_map), desc=f"{cat_key} extract", unit="user", leave=False, position=pos * 2 + 1) as pbar_users_ext:
                for uid, items in user_items_map.items():
                    extract_user_sync(uid, items, client, output_dir, error_records)
                    pbar_users_ext.update(1)

            # Write error logs (if any)
            if error_records:
                log_path = output_dir / "error_logs.txt"
                with open(log_path, "w", encoding="utf-8") as lf:
                    for rec in error_records:
                        lf.write(f"user_id={rec.get('user_id','')}\turl={rec.get('url','')}\terror={rec.get('error','')}\n")

            # Load checklists from search input file
            checklists = load_checklists(checklist_input_path)

            # Select same users, and filter to pending evaluation
            discovered = discover_extraction_files(output_dir, user_ids)
            filtered: List[Tuple[str, List[Path]]] = []
            for uid, files in discovered:
                need: List[Path] = []
                for fpath in files:
                    if not (fpath.parent / 'evaluation_result.json').exists():
                        need.append(fpath)
                if need:
                    filtered.append((uid, need))

            remaining_users_eval = [uid for uid, _ in filtered]
            print(f"[{cat_key}] Remaining users to evaluate: {len(remaining_users_eval)}" + (f" -> {', '.join(remaining_users_eval)}" if remaining_users_eval else ""))

            with tqdm(total=len(filtered), desc=f"{cat_key} eval", unit="user", leave=False, position=pos * 2 + 2) as pbar_users_eval:
                for uid, files in filtered:
                    evaluate_user(uid, files, checklists, client)
                    pbar_users_eval.update(1)

        with ThreadPoolExecutor(max_workers=len(categories) or 1) as cat_pool:
            cat_futs = {cat_pool.submit(process_category, cat, idx): cat for idx, cat in enumerate(categories)}
            for fut in as_completed(cat_futs):
                _ = cat_futs[fut]
                fut.result()
                pbar_cats.update(1)

    print("Pipeline completed for requested categories/users.")


if __name__ == '__main__':
    main()
