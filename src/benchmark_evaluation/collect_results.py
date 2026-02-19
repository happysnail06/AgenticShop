"""
Purpose:
    - Collect evaluation results across items for a given user and category.
    - Aggregate per-item stats and annotated checklists, and compute averages.
    - Support for grocery category variants: attribute_specific, brand_categorical, explicit.

Usage:
    - Run from repository root or any directory; provide base path to results.
    - Examples:
        # Process all users in a category (writes to each user directory)
        python -m src.collect_results \
            --category electronics \
            --base-path /work/AgenticShop/eval_results

        # Process grocery variant (variant is required for grocery)
        python -m src.collect_results \
            --category grocery \
            --variant attribute_specific \
            --base-path /work/AgenticShop/eval_results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


CATEGORY_MAPPING: Dict[str, str] = {
    'grocery': 'Grocery_and_Gourmet_Food',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'electronics': 'Electronics',
    'home': 'Home_and_Kitchen',
    'open': 'open_ended_curation',
}

# Grocery variants that require special handling
GROCERY_VARIANTS: Tuple[str, ...] = ('attribute_specific', 'brand_categorical', 'explicit')


X_OF_Y_FIELDS: Tuple[str, ...] = (
    'brand_preferences_percentage',
    'price_sensitivity_percentage',
    'review_sensitivity_percentage',
    'functional_requirements_percentage',
    'aesthetic_preferences_percentage',
    'purchase_preferences_percentage',
)

NUMERIC_FIELDS: Tuple[str, ...] = (
    'satisfaction_percentage',
    'satisfied_items',
    'not_satisfied_items',
    'total_checklist_items',
)


@dataclass
class ItemResult:
    item_id: str
    stats: Dict[str, Any]
    annotated_checklist: Dict[str, Any]


def parse_x_of_y(value: str) -> Tuple[int, int]:
    """Parse strings like "3/5" into (3, 5). If invalid, returns (0, 0)."""
    if isinstance(value, str) and '/' in value:
        numerator_str, denominator_str = value.split('/', 1)
        numerator_str = numerator_str.strip()
        denominator_str = denominator_str.strip()
        if numerator_str.isdigit() and denominator_str.isdigit():
            return int(numerator_str), int(denominator_str)
    return 0, 0


def format_x_of_y(numerator: int, denominator: int) -> str:
    """Format aggregated x/y values back into a "x/y" string."""
    return f"{numerator}/{denominator}"


def read_json_file(path: str) -> Optional[Dict[str, Any]]:
    """Read and decode JSON file if it exists; returns None if missing or empty."""
    if not os.path.exists(path) or not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
        if not raw:
            return None
        return json.loads(raw)


def find_item_result_dirs(user_dir: str) -> List[Tuple[str, str]]:
    """Find tuples of (item_id, evaluation_result.json path) under a user's results directory."""
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(user_dir):
        return pairs

    # Expect structure: user_{id}/result_{NNN}/evaluation_result.json
    for result_name in sorted(os.listdir(user_dir)):
        result_dir = os.path.join(user_dir, result_name)
        if not os.path.isdir(result_dir) or not result_name.startswith('result_'):
            continue
        eval_path = os.path.join(result_dir, 'evaluation_result.json')
        if os.path.isfile(eval_path):
            # Use result_name as item_id (e.g., "result_000" -> "result_000")
            pairs.append((result_name, eval_path))
    return pairs


def list_user_dirs(category_dir: str) -> List[str]:
    """List absolute paths to user directories under the category directory."""
    if not os.path.isdir(category_dir):
        return []
    paths: List[str] = []
    for name in sorted(os.listdir(category_dir)):
        path = os.path.join(category_dir, name)
        if os.path.isdir(path) and name.startswith('user_'):
            paths.append(path)
    return paths


def collect_results(base_path: str, category: str, user_id: str, variant: Optional[str] = None) -> Tuple[Dict[str, ItemResult], Dict[str, Any]]:
    """Collect item results for a user and compute average stats.

    Returns a mapping of item_id -> ItemResult and the computed average stats dict.
    """
    category_dir = get_category_dir(base_path, category, variant)
    user_dir = os.path.join(category_dir, f'user_{user_id}')

    collected: Dict[str, ItemResult] = {}

    # Aggregators
    numeric_sums: Dict[str, float] = {field: 0.0 for field in NUMERIC_FIELDS}
    xy_sums: Dict[str, Tuple[int, int]] = {field: (0, 0) for field in X_OF_Y_FIELDS}
    count: int = 0
    observed_total_checklist_items: Optional[int] = None

    for item_id, eval_path in find_item_result_dirs(user_dir):
        data = read_json_file(eval_path)
        if not data or 'evaluation_result' not in data:
            continue

        evaluation_result = data['evaluation_result']
        stats = evaluation_result.get('stats', {})
        annotated_checklist = evaluation_result.get('annotated_checklist', {})

        # Aggregate numeric fields
        for field in NUMERIC_FIELDS:
            value = stats.get(field)
            if isinstance(value, (int, float)):
                numeric_sums[field] += float(value)
                if field == 'total_checklist_items':
                    try:
                        v_int = int(value)
                        if observed_total_checklist_items is None:
                            observed_total_checklist_items = v_int
                    except Exception:
                        pass

        # Aggregate x/y fields
        for field in X_OF_Y_FIELDS:
            xy_value = stats.get(field)
            num, den = parse_x_of_y(xy_value)
            if num or den:
                cur_num, cur_den = xy_sums[field]
                xy_sums[field] = (cur_num + num, cur_den + den)

        collected[item_id] = ItemResult(
            item_id=item_id,
            stats=stats,
            annotated_checklist=annotated_checklist,
        )
        count += 1

    average_stats: Dict[str, Any] = {}
    if count > 0:
        for field in NUMERIC_FIELDS:
            if field == 'total_checklist_items':
                continue
            average_stats[f'average_{field}'] = numeric_sums[field] / float(count)
        # Use common per-item checklist length (not a sum)
        average_stats['total_checklist_items'] = int(observed_total_checklist_items or 0)

        for field in X_OF_Y_FIELDS:
            num, den = xy_sums[field]
            if den > 0:
                average_stats[f'average_{field}'] = (num / den) * 100.0
            else:
                average_stats[f'average_{field}'] = 0.0

        # Derived percentage fields for satisfied/not_satisfied over total_checklist_items
        total_per_item = float(observed_total_checklist_items or 0)
        total_for_all_items = total_per_item * float(count)
        if total_for_all_items > 0.0:
            average_stats['average_satisfied_items_percentage'] = (numeric_sums.get('satisfied_items', 0.0) / total_for_all_items) * 100.0
            average_stats['average_not_satisfied_items_percentage'] = (numeric_sums.get('not_satisfied_items', 0.0) / total_for_all_items) * 100.0
        else:
            average_stats['average_satisfied_items_percentage'] = 0.0
            average_stats['average_not_satisfied_items_percentage'] = 0.0

        # Round averages to 4 decimals, keep total_checklist_items as integer
        average_stats = round_numeric_fields(average_stats, ndigits=4, exclude_keys=['total_checklist_items'])

    return collected, average_stats


def merge_annotated_checklists(collected: Dict[str, ItemResult]) -> Dict[str, List[Dict[str, Any]]]:
    """Merge annotated_checklist across items, preserving category keys and tagging item_id."""
    merged: Dict[str, List[Dict[str, Any]]] = {}

    for item_id, result in collected.items():
        checklist = result.annotated_checklist or {}
        for category_key, entries in checklist.items():
            if not isinstance(entries, list):
                continue
            bucket = merged.setdefault(category_key, [])
            for entry in entries:
                if isinstance(entry, dict):
                    enriched = dict(entry)
                    enriched['item_id'] = item_id
                    bucket.append(enriched)
    return merged


def build_output_json(user_id: str, collected: Dict[str, ItemResult], average_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Build final JSON structure per specification."""
    collected_results: Dict[str, Any] = {}
    for item_id in sorted(collected.keys()):
        item = collected[item_id]
        collected_results[item_id] = {
            'stats': item.stats,
            'annotated_checklist': item.annotated_checklist,
        }

    # Do not aggregate annotated_checklist; only include averaged stats at the user level
    user_evaluation_result = {
        'user_id': user_id,
        'stats': average_stats,
        'collected_results': collected_results,
    }

    return {
        'user_evaluation_result': user_evaluation_result,
        'average_stats': average_stats,
    }


def get_category_dir(base_path: str, category: str, variant: Optional[str] = None) -> str:
    category_dir_name = CATEGORY_MAPPING.get(category.lower(), category)
    category_dir = os.path.join(base_path, category_dir_name)
    
    # For grocery category, add variant subdirectory if specified
    if category.lower() == 'grocery' and variant and variant in GROCERY_VARIANTS:
        category_dir = os.path.join(category_dir, variant)
    
    return category_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect and aggregate evaluation results for users.')
    parser.add_argument('--category', required=True, help='One of: grocery, clothing, electronics, home, open')
    parser.add_argument('--variant', required=False, help='For grocery category: attribute_specific, brand_categorical, or explicit')
    parser.add_argument('--user-id', required=False, help='If provided, process only this user (suffix without "user_")')
    default_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'eval_results')
    parser.add_argument('--base-path', default=default_base, help='Base path to evaluation results root')
    parser.add_argument('--output', required=False, help='Optional explicit output path (only when processing a single user)')
    return parser.parse_args()


def write_user_summary(output_obj: Dict[str, Any], output_path: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)


def round_numeric_fields(d: Dict[str, Any], ndigits: int = 4, exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    exclude = set(exclude_keys or [])
    rounded: Dict[str, Any] = {}
    for k, v in d.items():
        if k in exclude:
            rounded[k] = v
            continue
        if isinstance(v, (int, float)):
            rounded[k] = round(float(v), ndigits)
        else:
            rounded[k] = v
    return rounded


def main() -> None:
    args = parse_args()

    # Validate variant for grocery category
    if args.category.lower() == 'grocery' and args.variant and args.variant not in GROCERY_VARIANTS:
        print(f"Error: Invalid variant '{args.variant}' for grocery category. Must be one of: {', '.join(GROCERY_VARIANTS)}")
        sys.exit(1)
    
    # Require variant for grocery category
    if args.category.lower() == 'grocery' and not args.variant:
        print(f"Error: Variant is required for grocery category. Must be one of: {', '.join(GROCERY_VARIANTS)}")
        sys.exit(1)

    category_dir = get_category_dir(args.base_path, args.category, args.variant)

    if args.user_id:
        collected, average_stats = collect_results(
            base_path=args.base_path,
            category=args.category,
            user_id=args.user_id,
            variant=args.variant,
        )
        output_obj = build_output_json(user_id=args.user_id, collected=collected, average_stats=average_stats)
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(category_dir, f'user_{args.user_id}', 'evaluation_summary.json')
        write_user_summary(output_obj, output_path)
        return

    # Process all users in the category
    user_stats_sums: Dict[str, float] = {}
    users_count: int = 0
    observed_total_checklist_items_all_users: Optional[int] = None

    for user_path in list_user_dirs(category_dir):
        user_name = os.path.basename(user_path)
        if not user_name.startswith('user_'):
            continue
        user_id = user_name[len('user_'):]
        collected, average_stats = collect_results(
            base_path=args.base_path,
            category=args.category,
            user_id=user_id,
            variant=args.variant,
        )
        output_obj = build_output_json(user_id=user_id, collected=collected, average_stats=average_stats)
        output_path = os.path.join(user_path, 'evaluation_summary.json')
        write_user_summary(output_obj, output_path)

        # Accumulate per-user averaged stats for cross-user aggregation
        if isinstance(average_stats, dict) and average_stats:
            for k, v in average_stats.items():
                if k == 'total_checklist_items':
                    try:
                        v_int = int(v)
                        if observed_total_checklist_items_all_users is None:
                            observed_total_checklist_items_all_users = v_int
                    except Exception:
                        pass
                    continue
                if isinstance(v, (int, float)):
                    user_stats_sums[k] = user_stats_sums.get(k, 0.0) + float(v)
        users_count += 1

    # Write cross-user averaged stats at category level
    if users_count > 0 and user_stats_sums:
        cross_user_avg_stats: Dict[str, Any] = {}
        for k, sum_v in user_stats_sums.items():
            cross_user_avg_stats[k] = sum_v / float(users_count)
        # Preserve non-averaged total_checklist_items as the common value
        cross_user_avg_stats['total_checklist_items'] = int(observed_total_checklist_items_all_users or 0)

        # Round cross-user averages to 4 decimals, keep total_checklist_items as integer
        cross_user_avg_stats = round_numeric_fields(cross_user_avg_stats, ndigits=4, exclude_keys=['total_checklist_items'])

        category_output_path = os.path.join(category_dir, 'evaluation_results.json')
        write_user_summary({'stats': cross_user_avg_stats}, category_output_path)


if __name__ == '__main__':
    main()
