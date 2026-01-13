#!/usr/bin/env python3
"""
Unified pipeline: Extraction → Evaluation

Purpose: 
    - Runs extraction for selected categories/users
    - Then runs evaluation for those outputs
    - Parallelizes across categories; processes users sequentially; items per user run concurrently

Usage:
    python run_pipeline.py \
      --model-type search_llms \
      --model-name gpt \
      --category clothing \
      --num-users 1 \
      [--all-users] [--all-categories]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from benchmark_evaluation.modules import Extractor, Evaluator


CATEGORY_MAPPING: Dict[str, str] = {
    'grocery': 'Grocery_and_Gourmet_Food',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'electronics': 'Electronics',
    'home': 'Home_and_Kitchen',
    'open': 'open_ended_curation',
}


def get_project_paths():
    """Get project root and other important paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return {
        "project_root": project_root,
        "env_path": os.path.join(project_root, ".env.local"),
        "input_dir": os.path.join(project_root, "eval_inputs"),
        "checklist_dir": os.path.join(project_root, "eval_inputs", "_user_checklists"),
        "output_dir": os.path.join(project_root, "eval_results")
    }


def build_paths(model_type: str, model_name: str, category_key: str) -> tuple[Path, Path, str, str]:
    """Build input/output paths and metadata for a given category.

    Args:
        model_type (str): Model family, e.g., 'web_agents' or 'search_llms'.
        model_name (str): Specific model identifier used in path layout.
        category_key (str): Short category key (e.g., 'clothing', 'grocery_attribute').

    Returns:
        tuple[Path, Path, str, str]: (input_path, output_dir, category_full_name, variant).
    """
    input_base = Path(f"/work/AgenticShop/eval_inputs/{model_type}/{model_name}")
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
        filename = grocery_files[category_key]
        category_full = CATEGORY_MAPPING['grocery']
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


def build_output_base(model_type: str, model_name: str, category_key: str) -> Path:
    """Return the base output directory for a given category and model.

    Args:
        model_type (str): Model family name.
        model_name (str): Specific model identifier.
        category_key (str): Category key which may imply a variant.

    Returns:
        Path: Base directory under which user results are written.
    """
    output_root = Path(f"/work/AgenticShop/eval_results/{model_type}/{model_name}")
    if category_key in ('clothing', 'electronics', 'home', 'open'):
        return output_root / CATEGORY_MAPPING[category_key]
    variants = {
        'grocery_attribute': 'attribute_specific',
        'grocery_brand': 'brand_categorical',
        'grocery_explicit': 'explicit',
    }
    return output_root / CATEGORY_MAPPING['grocery'] / variants[category_key]


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
    import glob
    extraction_files = glob.glob(pattern)
    for file_path in extraction_files:
        path_obj = Path(file_path)
        user_dir = path_obj.parent.parent.name
        user_id = user_dir[5:]
        if user_id in user_ids:
            if user_id not in user_extractions_map:
                user_extractions_map[user_id] = []
            user_extractions_map[user_id].append(path_obj)
    for uid in user_extractions_map:
        user_extractions_map[uid].sort()
    return list(user_extractions_map.items())


async def extract_user_items(user_id: str, items: List[Dict[str, Any]], extractor: Extractor, base_out: Path,
                             error_records: List[Dict[str, Any]] | None = None) -> None:
    """Run extraction concurrently for a user's pending items.

    Args:
        user_id (str): Target user identifier.
        items (List[Dict[str, Any]]): Minimal item payloads (e.g., product URLs).
        extractor (Extractor): Extraction module instance.
        base_out (Path): Base output directory for the category/variant.
    """
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
    """Synchronous wrapper to run the async extraction for a user.

    Args:
        user_id (str): Target user identifier.
        items (List[Dict[str, Any]]): Pending items for extraction.
        client (OpenAI): OpenAI client used by the extractor.
        base_out (Path): Base output directory.
    """
    extractor = Extractor(client)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(extract_user_items(user_id, items, extractor, base_out, error_records))
    finally:
        # Cancel and drain any leftover tasks to avoid "Task was destroyed but it is pending!"
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        # Shutdown async generators (and default executor where available)
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
        checklists (Dict[str, Any]): Loaded per-user checklists for the category.
        client (OpenAI): OpenAI client used by the evaluator.
    """
    evaluator = Evaluator(client)
    tasks: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
    check_list = checklists[f"user_{user_id}"]["check_list"]
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

    # Category progress
    with tqdm(total=len(categories), desc="Categories", unit="cat") as pbar_cats:
        def process_category(cat_key: str, pos: int) -> None:
            input_path, output_dir, _, _ = build_paths(args.model_type, args.model_name, cat_key)
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(input_path, 'r', encoding='utf-8') as f:
                eval_input: Dict[str, Any] = json.load(f)
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
            # Print remaining users pending extraction for this category
            remaining_users_extract = list(user_items_map.keys())
            print(f"[{cat_key}] Remaining users to extract: {len(remaining_users_extract)}" + (f" -> {', '.join(remaining_users_extract)}" if remaining_users_extract else ""))
            # print(f"[{cat_key}] Remaining users to extract: {len(remaining_users_extract)}")
            
            # Collect errors during user processing
            # Extract sequentially per user with items concurrent within user
            error_records: List[Dict[str, Any]] = []
            with tqdm(total=len(user_items_map), desc=f"{cat_key} extract", unit="user", leave=False, position=pos * 2 + 1) as pbar_users_ext:
                for uid, items in user_items_map.items():
                    extract_user_sync(uid, items, client, output_dir, error_records)
                    pbar_users_ext.update(1)

            # Write error logs (if any) under the category directory
            if error_records:
                log_path = output_dir / "error_logs.txt"
                with open(log_path, "w", encoding="utf-8") as lf:
                    for rec in error_records:
                        lf.write(f"user_id={rec.get('user_id','')}\turl={rec.get('url','')}\terror={rec.get('error','')}\n")

            # Evaluation for this category
            checklist_root = Path(paths["checklist_dir"])
            checklist_file = checklist_root / f"{(CATEGORY_MAPPING[cat_key] if cat_key in ('clothing','electronics','home','open') else CATEGORY_MAPPING['grocery'])}_check_list.json"
            with open(checklist_file, 'r', encoding='utf-8') as f:
                checklists: Dict[str, Any] = json.load(f)

            base_dir = output_dir
            # Select same users, and filter to pending evaluation
            discovered = discover_extraction_files(base_dir, user_ids)
            filtered: List[Tuple[str, List[Path]]] = []
            for uid, files in discovered:
                need: List[Path] = []
                for fpath in files:
                    if not (fpath.parent / 'evaluation_result.json').exists():
                        need.append(fpath)
                if need:
                    filtered.append((uid, need))
            # Print remaining users pending evaluation for this category
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
