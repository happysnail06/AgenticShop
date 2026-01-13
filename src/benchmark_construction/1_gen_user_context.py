#!/usr/bin/env python3
"""
User Context Generation:
    Generate user context for each user based on their purchase history.
    
    Usage example:
        python src/benchmark_construction/1_gen_user_context.py --domain clothing --samples 3
        
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
from typing import Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompts.user_context_gen_prompt import build_user_context_gen_prompt, build_open_user_context_gen_prompt


class UserContext(BaseModel):
    brand_preferences: str
    price_sensitivity: str
    review_sensitivity: str
    functional_requirements: str
    aesthetic_preferences: str
    purchase_preferences: str
    
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
        "input_dir": os.path.join(project_root, "data", "user_raw"),
        "output_dir": os.path.join(project_root, "data", "user_staging", "context")
    }

async def generate_user_context(runnable, prompt: str) -> UserContext:
    result = await runnable.ainvoke({"input": prompt})
    return result

async def process_user(user_item, user_context_runnable, semaphore, domain):
    """Process a single user's context generation.
    
    Args:
        user_item: Tuple of (user_id, item_info)
        user_context_runnable: The runnable for generating user context
        semaphore: Semaphore for rate limiting
        domain: Domain being processed (to determine review count)
        
    Returns:
        Dict with user and user_context, or None if failed
    """
    user, item_info = user_item
    async with semaphore:
        try:
            # Use all reviews for open domain, first 10 for others
            if domain == 'open_curation':
                purchase_history = item_info[1]['review']  # Use all 20 reviews
                instruction = build_open_user_context_gen_prompt(purchase_history=purchase_history)
            else:
                purchase_history = item_info[1]['review'][:10]  # Use first 10 reviews
                instruction = build_user_context_gen_prompt(purchase_history=purchase_history)
            
            prompt = instruction
            
            user_context_obj = await generate_user_context(user_context_runnable, prompt)
            user_context = user_context_obj.model_dump()
            
            item_info[1]['user_context'] = user_context
            
            return {
                "user": user,
                "user_context": user_context,
            }
        except Exception as e:
            print(f"Error generating user context for user {user}: {e}")
            return None

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="clothing", help="Domain short key, e.g., clothing, grocery, electronics, home, open")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to process (default: all)")
    return parser.parse_args()

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
        reasoning={"effort": "medium"},
        verbosity="medium",
    )

    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates comprehensive user profiles. Return only valid JSON for the specified schema."),
        ("human", "{input}")
    ])

    def build_runnable():
        return prompt_tmpl | llm.with_structured_output(UserContext)

    # Resolve domain via mapping; allow full value passthrough if already full
    mapped_domain = CATEGORY_MAPPING.get(args.domain.lower(), args.domain)

    # Construct absolute path to data file
    if mapped_domain == 'open_curation':
        filename = "open_curation.json"
    else:
        filename = f"{mapped_domain}.json"
    
    data_path = os.path.join(paths["input_dir"], filename)
    data = json.load(open(data_path, 'r'))
    items = list(data.items())
    if args.samples is not None:
        items = items[:max(0, int(args.samples))]
    result = []
    total_len = len(items)

    user_context_runnable = build_runnable()

    # Process users concurrently with rate limiting
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
    
    # Create tasks for all users
    tasks = [process_user(user_item, user_context_runnable, semaphore, mapped_domain) for user_item in items]
    
    # Process with progress bar
    result = []
    with tqdm(total=len(tasks), desc="Generating user context") as pbar:
        for task in asyncio.as_completed(tasks):
            try:
                user_result = await task
                result.append(user_result)
                
            except Exception as e:
                print(f"Unexpected error: {e}")
            finally:
                pbar.update(1)

    # Create output directory if it doesn't exist
    os.makedirs(paths["output_dir"], exist_ok=True)
    
    output_path = os.path.join(paths["output_dir"], f"{args.domain}_user_context.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    asyncio.run(async_main())
