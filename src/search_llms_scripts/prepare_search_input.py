"""
Search Input Preparation:
    Build search prompts from user profile JSON entries for search LLM scripts.

    Usage example:
        python src/search_llms_scripts/prepare_search_input.py data/user_profiles/clothing_user_profile.json
        python src/search_llms_scripts/prepare_search_input.py data/user_profiles/electronics_user_profile.json
        python src/search_llms_scripts/prepare_search_input.py data/user_profiles/grocery_attribute_specific_user_profile.json
        python src/search_llms_scripts/prepare_search_input.py data/user_profiles/grocery_brand_categorical_user_profile.json
        python src/search_llms_scripts/prepare_search_input.py data/user_profiles/grocery_explicit_title_user_profile.json
        python src/search_llms_scripts/prepare_search_input.py data/user_profiles/home_user_profile.json
        python src/search_llms_scripts/prepare_search_input.py data/user_profiles/open_user_profile.json

    Output:
        Writes a JSON array to product_curation_artifacts/inputs/{domain}_search_input.json
        where {domain} is derived from the input filename prefix before "_user_profile.json".
"""

import json
import os
import sys
from typing import Any, Dict, List


PROMPT_TEMPLATE = (
    "You are a shopping assistant helping a user find products that fit their context and needs. "
)


def get_project_paths():
    """Get project root and other important paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return {
        "project_root": project_root,
        "output_dir": os.path.join(project_root, "product_curation_artifacts", "inputs"),
    }


def derive_domain_from_path(input_path: str) -> str:
    """Derive domain name from input filename like "clothing_user_profile.json" -> "clothing".

    Args:
        input_path (str): Path to the input user profiles JSON file.

    Returns:
        str: Domain name to use in output filename.
    """
    filename = os.path.basename(input_path)
    if filename.endswith("_user_profile.json"):
        return filename[: -len("_user_profile.json")]
    if filename.endswith(".json"):
        return filename[: -len(".json")]
    return filename


def build_variables(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract variables for prompt formatting from a profile entry.

    Args:
        entry (Dict[str, Any]): One user profile record.

    Returns:
        Dict[str, Any]: Mapping for PROMPT_TEMPLATE.format(**vars).
    """
    uc = entry.get("user_context", {})
    variables: Dict[str, Any] = {
        "seed_product_type": entry.get("seed_product_type", ""),
        "situation": entry.get("situation", ""),
        "budget_range": entry.get("budget_range", ""),
        "user_query": entry.get("user_query", ""),
        "brand_preferences": uc.get("brand_preferences", ""),
        "price_sensitivity": uc.get("price_sensitivity", ""),
        "review_sensitivity": uc.get("review_sensitivity", ""),
        "functional_requirements": uc.get("functional_requirements", ""),
        "aesthetic_preferences": uc.get("aesthetic_preferences", ""),
        "purchase_preferences": uc.get("purchase_preferences", ""),
    }
    return variables


def build_prompt_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Create a single model input object with template, variables, and formatted prompt.

    Args:
        entry (Dict[str, Any]): User profile record.

    Returns:
        Dict[str, Any]: Object containing identifiers, template, variables, and prompt string.
    """
    variables = build_variables(entry)

    # Build prompt conditionally, omitting lines with missing/empty values
    parts: List[str] = []
    parts.append(PROMPT_TEMPLATE)

    seed_product_type = variables.get("seed_product_type")
    if seed_product_type:
        parts.append(f"The user is looking for {seed_product_type}. ")

    situation = variables.get("situation")
    budget_range = variables.get("budget_range")
    user_query = variables.get("user_query")
    situation_line_parts: List[str] = []
    if situation:
        situation_line_parts.append(f"The user's situation is: {situation.rstrip('.')}.")
    if budget_range:
        situation_line_parts.append(f"They are thinking a budget {budget_range}.")
    if user_query:
        situation_line_parts.append(f"The user says: \"{user_query}\".")
    if situation_line_parts:
        parts.append(" ".join(situation_line_parts) + " ")

    # Context section
    context_keys = [
        "brand_preferences",
        "price_sensitivity",
        "review_sensitivity",
        "functional_requirements",
        "aesthetic_preferences",
        "purchase_preferences",
    ]
    context_present = [k for k in context_keys if variables.get(k)]
    if context_present:
        parts.append("The user's shopping preferences and context are as follows: ")
        for k in context_keys:
            v = variables.get(k)
            if v:
                escaped = v.replace('"', "'")
                parts.append(f"{k}: \"{escaped}\". ")

    parts.append(
        "Task: search online and recommend the top 3 products among your search results. "
        "For each pick, briefly explain why it fits this user, referencing their budget, needs, and preferences. "
        "The URL should be a direct link to the product selling page."
    )

    prompt = "".join(parts)
    return {
        "user": entry.get("user", ""),
        "seed_product_type": variables["seed_product_type"],
        "prompt": prompt,
        "variables": variables,
        "check_list": entry.get("check_list"),
    }


def main() -> None:
    """Entry point: read profiles, build prompts, write batchable JSON array."""
    if len(sys.argv) < 2:
        print(
            "Usage: python src/search_llms_scripts/prepare_search_input.py <path-to-user_profile.json>",
        )
        return

    paths = get_project_paths()
    input_path = sys.argv[1]
    domain = derive_domain_from_path(input_path)
    output_dir = paths["output_dir"]
    output_path = os.path.join(output_dir, f"{domain}_search_input.json")

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        profiles: List[Dict[str, Any]] = json.load(f)

    search_inputs: List[Dict[str, Any]] = [build_prompt_entry(e) for e in profiles]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(search_inputs, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(search_inputs)} entries to {output_path}")


if __name__ == "__main__":
    main()
