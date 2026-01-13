#!/usr/bin/env python3
"""
User Checklist Generation:
    Generate user checklists for each user based on their purchase history.
    
    Usage example:
        python src/benchmark_construction/3_gen_user_checklist.py --domain clothing --samples 3
        
    Domains:
        grocery: Grocery and Gourmet Food
        clothing: Clothing, Shoes and Jewelry
        electronics: Electronics
        home: Home and Kitchen
        open: Open-ended curation
"""
import argparse
from tqdm import tqdm
import pandas as pd 
import json
import os
from typing import List, Dict
from dotenv import load_dotenv
import asyncio
from typing import Optional, Type
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts.user_checklist_gen_prompt import build_user_checklist_prompt


class UserChecklist(BaseModel):
    brand_preferences: List[str]
    price_sensitivity: List[str]
    review_sensitivity: List[str]
    functional_requirements: List[str]
    aesthetic_preferences: List[str]
    purchase_preferences: List[str]

# Query type mapping for different domains
QUERY_TYPE_MAPPING = {
    'clothing': 'categorical',
    'electronics': 'categorical', 
    'home': 'categorical',
    'grocery': 'target_finding',  # Will generate three separate files
    'open': 'generic'             # Will use generic query
}


async def generate_user_checklist(runnable, prompt: str) -> UserChecklist:
    result: UserChecklist = await runnable.ainvoke({"input": prompt})
    return result

CATEGORY_MAPPING: Dict[str, str] = {
    'grocery': 'Grocery_and_Gourmet_Food',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'electronics': 'Electronics',
    'home': 'Home_and_Kitchen',
    'open': 'open_curation',
}

def get_project_paths():
    """Get project root and other important paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return {
        "project_root": project_root,
        "env_path": os.path.join(project_root, ".env.local"),
        "user_query_dir": os.path.join(project_root, "data", "_user_data", "_user_query"),
        "output_dir": os.path.join(project_root, "data", "_user_data", "_user_checklist")
    }

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="clothing", help="Domain short key, e.g., clothing, grocery, electronics, home, open")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to process (default: all)")
    return parser.parse_args()

async def process_user(user_item, user_checklist_runnable, semaphore, domain, query_type=None):
    """Process a single user's checklist generation.
    
    Args:
        user_item: Dictionary containing user data from _user_query
        user_checklist_runnable: The runnable for generating user checklists
        semaphore: Semaphore for rate limiting
        domain: The domain being processed
        query_type: Specific query type for grocery domain
        
    Returns:
        Dict with user and generated checklist, or None if failed
    """
    async with semaphore:
        try:
            user_context = user_item["user_context"]
            user_situation = user_item["situation"]
            user_budget = user_item["budget_range"]
            
            # Determine which query to use based on domain and query_type
            if domain == 'grocery' and query_type:
                # For grocery, use specific query type
                user_query = user_item["user_query"][query_type]
            elif domain == 'open':
                # For open, use generic query
                user_query = "curate a list of products that i would be interested based on my profile from the web."
            else:
                # For clothing, electronics, home - use categorical query
                user_query = user_item["user_query"]["categorical"]
            
            # Build the prompt with user context, situation, and budget
            instruction = build_user_checklist_prompt(
                user_context=user_context,
                user_situation=user_situation,
                user_budget=user_budget
            )
            prompt = instruction
            
            
            user_checklist_obj = await generate_user_checklist(user_checklist_runnable, prompt)
            user_checklist = user_checklist_obj.model_dump()
            
            return {
                "user": user_item["user"],
                "seed_product_type": user_item["seed_product_type"],
                "situation": user_item["situation"],
                "budget_range": user_budget,
                "user_query": user_query,
                "user_context": user_context,
                "check_list": user_checklist,
                "seed_product_information": user_item["seed_product_information"],
            }
        except Exception as e:
            print(f"Error generating checklist for user {user_item.get('user', 'unknown')}: {e}")
            return None

async def async_main():
    # Setup paths and environment
    paths = get_project_paths()
    load_dotenv(paths["env_path"])
    args = argparser()

    # Build model and runnable factory
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided.")

    llm = ChatOpenAI(
        model="gpt-5-mini",
        output_version="responses/v1",
        api_key=api_key,
        reasoning={"effort": "low"},
        verbosity="low",
    )
    # llm = ChatOpenAI(
    #     model="gpt-4.1",
    #     api_key=api_key,
    #     temperature=0.5,
    # )
    
    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "Return only valid JSON for the specified schema."),
        ("human", "{input}")
    ])

    def build_runnable():
        return prompt_tmpl | llm.with_structured_output(UserChecklist)

    # Load user query data
    user_query_path = os.path.join(paths["user_query_dir"], f"{args.domain}_user_query.json")
    user_query_data = json.load(open(user_query_path, 'r'))
    
    items = user_query_data
    if args.samples is not None:
        items = items[:max(0, int(args.samples))]
    
    user_checklist_runnable = build_runnable()
    query_type = QUERY_TYPE_MAPPING.get(args.domain, 'categorical')
    
    # Create output directory if it doesn't exist
    os.makedirs(paths["output_dir"], exist_ok=True)
    
    if query_type == 'categorical' or query_type == 'generic':
        # For clothing, electronics, home, open - single file
        semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
        
        # Create tasks for all users
        tasks = [process_user(user_item, user_checklist_runnable, semaphore, args.domain) for user_item in items]
        
        # Process with progress bar
        result = []
        with tqdm(total=len(tasks), desc="Generating user checklists") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    user_result = await task
                    if user_result is not None:
                        result.append(user_result)
                except Exception as e:
                    print(f"Unexpected error: {e}")
                finally:
                    pbar.update(1)
        
        # Save single file
        output_path = os.path.join(paths["output_dir"], f"{args.domain}_user_checklist.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
            
    elif query_type == 'target_finding':
        # For grocery - generate once, write three variation files with same checklist
        semaphore = asyncio.Semaphore(10)
        tasks = [process_user(user_item, user_checklist_runnable, semaphore, args.domain, 'categorical') for user_item in items]
        # We still compute checklists independent of query wording
        unified_result = []
        with tqdm(total=len(tasks), desc="Generating grocery checklists") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    user_result = await task
                    if user_result is not None:
                        unified_result.append(user_result)
                except Exception as e:
                    print(f"Unexpected error: {e}")
                finally:
                    pbar.update(1)

        # Now write three files, adapting only the user_query string per variation from original items
        query_variations = ['explicit_title', 'attribute_specific', 'brand_categorical']
        for variation in query_variations:
            adapted = []
            for idx, base in enumerate(unified_result):
                # Replace user_query with the specific variation from source items
                src = items[idx]
                new_item = dict(base)
                new_item["user_query"] = src["user_query"][variation]
                adapted.append(new_item)
            output_path = os.path.join(paths["output_dir"], f"{args.domain}_{variation}_user_checklist.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(adapted, f, indent=4)


if __name__ == "__main__":
    asyncio.run(async_main())
        