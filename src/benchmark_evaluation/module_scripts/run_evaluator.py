#!/usr/bin/env python3
"""
Evaluation script for check_list satisfaction assessment.

Purpose: 
    - Evaluates extraction_result.json files (model responses) against user check_list
    - Saves evaluation_result.json next to each extraction_result.json
    - Concurrency model: categories concurrent; users sequential per category

Usage:
    python run_evaluator.py \
      --model-type search_llms \
      --model-name gpt \
      --category clothing \
      --num-users 1 \
      [--all-users] [--all-categories]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
# Category mapping and helpers aligned with run_extractor
from typing import Dict
from dotenv import load_dotenv
from openai import OpenAI

# Ensure the src directory is on sys.path for direct execution
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from benchmark_evaluation.modules import Evaluator


CATEGORY_MAPPING: Dict[str, str] = {
    'grocery': 'Grocery_and_Gourmet_Food',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'electronics': 'Electronics',
    'home': 'Home_and_Kitchen',
    'open': 'open_ended_curation',
}

def build_output_base(model_type: str, model_name: str, category_key: str) -> Path:
    """Return the base dir that contains user_* folders for a category/variant."""
    output_root = Path(f"/work/AgenticShop/eval_results/{model_type}/{model_name}")

    grocery_variants = {
        'grocery_attribute': 'attribute_specific',
        'grocery_brand': 'brand_categorical',
        'grocery_explicit': 'explicit',
    }

    if category_key in ('clothing', 'electronics', 'home', 'open'):
        category_full = CATEGORY_MAPPING[category_key]
        return output_root / category_full
    else:
        category_full = CATEGORY_MAPPING['grocery']
        variant = grocery_variants[category_key]
        return (output_root / category_full / variant)


def discover_extraction_files(base_dir: Path, user_ids: List[str] = None) -> List[Tuple[str, List[Path]]]:
    """Discover existing extraction_result.json files for evaluation.
    
    Args:
        base_dir (Path): Base directory containing user_* folders.
        user_ids (List[str], optional): Specific user IDs to process. If None, process all found users.
        
    Returns:
        List[Tuple[str, List[Path]]]: List of (user_id, extraction_file_paths) tuples.
    """
    user_extractions_map: Dict[str, List[Path]] = {}
    pattern = str(base_dir / "user_*" / "result_*" / "extraction_result.json")
    extraction_files = glob.glob(pattern)
    for file_path in extraction_files:
        path_obj = Path(file_path)
        user_dir = path_obj.parent.parent.name  # user_{user_id}
        user_id = user_dir[5:]
        if user_ids is None or user_id in user_ids:
            if user_id not in user_extractions_map:
                user_extractions_map[user_id] = []
            user_extractions_map[user_id].append(path_obj)
    for uid in user_extractions_map:
        user_extractions_map[uid].sort()
    return list(user_extractions_map.items())


def load_user_check_list(checklists: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Load user check_list from an in-memory checklist mapping."""
    return checklists[f"user_{user_id}"]["check_list"]


def evaluate_user_extractions(user_id: str, extraction_files: List[Path], 
                              checklists: Dict[str, Any], client: OpenAI) -> None:
    """Evaluate all extraction files for a single user against their check_list.
    
    Args:
        user_id (str): User identifier.
        extraction_files (List[Path]): List of extraction_result.json file paths for this user.
        base_dir (Path): Base evaluation results directory.
        client (OpenAI): OpenAI client instance.
    """
    # print(f"Evaluating user {user_id} with {len(extraction_files)} extraction files...")
    check_list = load_user_check_list(checklists, user_id)
    evaluator = Evaluator(client)

    # Build tasks: (file_path, model_data, check_list)
    evaluation_tasks: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []
    for extraction_file in extraction_files:
        with open(extraction_file, 'r', encoding='utf-8') as f:
            model_response = json.load(f)
        model_data = { 'search_output': model_response }
        evaluation_tasks.append((extraction_file, model_data, check_list))

    evaluation_reports = evaluator.evaluate_batch(evaluation_tasks, max_workers=len(evaluation_tasks))

    # Save each evaluation next to its extraction_result.json
    for i, (extraction_file, _, __) in enumerate(evaluation_tasks):
        evaluation_file = extraction_file.parent / 'evaluation_result.json'
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_reports[i], f, ensure_ascii=False, indent=2)


def main() -> None:
    """Main function for evaluation processing."""
    parser = argparse.ArgumentParser(description="Run evaluation on extraction results")
    parser.add_argument('--model-type', type=str, required=True, choices=['web_agents','search_llms'], help='Model type (web_agents or search_llms)')
    parser.add_argument('--model-name', type=str, required=True, help='Model name key (e.g., gpt)')
    parser.add_argument('--category', type=str,
                       choices=['clothing','electronics','home','open','grocery_attribute','grocery_brand','grocery_explicit'],
                       help='Category key or grocery variant')
    parser.add_argument('--num-users', type=int, default=1, 
                       help='Number of user folders to evaluate (first N by name)')
    parser.add_argument('--all-users', action='store_true', help='Evaluate all users within each category')
    parser.add_argument('--all-categories', action='store_true', help='Evaluate all supported categories for the model')
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
    categories_to_process = supported_categories if args.all_categories else [args.category]

    def build_checklist_path(cat_key: str) -> Path:
        output_root = Path("/work/agent-as-judge/data/evaluation/_user_checklists")
        if cat_key in ('clothing', 'electronics', 'home', 'open'):
            category_full = CATEGORY_MAPPING[cat_key]
        else:
            category_full = CATEGORY_MAPPING['grocery']
        return output_root / f"{category_full}_check_list.json"

    def process_category(cat_key: str) -> None:
        base_dir = build_output_base(args.model_type, args.model_name, cat_key)
        user_dirs = sorted([p for p in base_dir.glob('user_*') if p.is_dir()])
        all_user_ids = [p.name[5:] for p in user_dirs]
        user_ids_to_process = all_user_ids if args.all_users else all_user_ids[: max(1, args.num_users)]

        user_extraction_files = discover_extraction_files(base_dir, user_ids_to_process)
        # Filter out already evaluated items (existing evaluation_result.json)
        filtered: List[Tuple[str, List[Path]]] = []
        for uid, files in user_extraction_files:
            pending_files: List[Path] = []
            for fpath in files:
                eval_path = fpath.parent / 'evaluation_result.json'
                if not eval_path.exists():
                    pending_files.append(fpath)
            if pending_files:
                filtered.append((uid, pending_files))
        user_extraction_files = filtered
        # print(f"[{cat_key}] Found {len(user_extraction_files)} users with extraction results")

        # Load checklists file for this category once
        checklist_path = build_checklist_path(cat_key)
        with open(checklist_path, 'r', encoding='utf-8') as f:
            checklists: Dict[str, Any] = json.load(f)

        # Process users sequentially within the category
        for user_id, extraction_files in user_extraction_files:
            evaluate_user_extractions(user_id, extraction_files, checklists, client)

    # Run categories in parallel
    with ThreadPoolExecutor(max_workers=len(categories_to_process) or 1) as cat_pool:
        cat_futures = {cat_pool.submit(process_category, cat_key): cat_key for cat_key in categories_to_process}
        for fut in as_completed(cat_futures):
            cat_key = cat_futures[fut]
            fut.result()
            print(f"✓ Category evaluation done: {cat_key}")

    print("All evaluation tasks completed across requested categories!")


if __name__ == '__main__':
    main()
