#!/usr/bin/env python3
"""
User Query Generation:
    Generate user queries for each user based on their purchase history.
    
    Usage example:
        python src/benchmark_construction/2_gen_user_query.py --domain clothing --samples 3
        
    Domains:
        grocery: Grocery and Gourmet Food
        clothing: Clothing, Shoes and Jewelry
        electronics: Electronics
        home: Home and Kitchen
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
from prompts.user_query_gen_prompt import build_user_query_gen_prompt


class QueryDetails(BaseModel):
    explicit_title: str
    attribute_specific: str
    brand_categorical: str
    categorical: str

class UserQuery(BaseModel):
    situation: str
    budget_range: str
    query: QueryDetails

# Domain mapping for query types
QUERY_TYPE_MAPPING = {
    'clothing': 'categorical',
    'electronics': 'categorical', 
    'home': 'categorical',
    'grocery': 'all_variations',  # Will generate all three variations
    'open': 'generic'      # Will use generic query
    }
    
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
        "raw_data_dir": os.path.join(project_root, "data", "user_raw"),
        "user_context_dir": os.path.join(project_root, "data", "user_staging", "context"),
        "output_dir": os.path.join(project_root, "data", "user_staging", "query")
    }

async def generate_user_query(runnable, prompt: str) -> UserQuery:
    result = await runnable.ainvoke({"input": prompt})
    return result

async def process_user(user_item, user_query_runnable, semaphore, user_context_data, domain):
    """Process a single user's query generation.
    
    Args:
        user_item: Tuple of (user_id, item_info)
        user_query_runnable: The runnable for generating user queries
        semaphore: Semaphore for rate limiting
        user_context_data: Dictionary containing user context data
        domain: Domain being processed
        
    Returns:
        Dict with user and generated queries, or None if failed
    """
    user, item_info = user_item
    async with semaphore:
        try:
            # Get user context from the loaded context data
            user_context = user_context_data.get(user).get("user_context")
            
            # Handle open domain differently - no API call needed
            if domain == 'open':
                return {
                    "user": user,
                    "seed_product_type": None,
                    "user_context": user_context,
                    "situation": "I’m just exploring and seeing if there are any interesting products around.",
                    "budget_range": None,
                    "user_query": {
                        "explicit_title": None,
                        "attribute_specific": None,
                        "brand_categorical": None,
                        "categorical": None
                    },
                    "seed_product_information": None,
                }
            else:
                # Get seed product information from the last entry
                seed_product_info = item_info[1]['target']
                
                # Build the prompt with user context and seed product info
                instruction = build_user_query_gen_prompt(
                    user_context=user_context,
                    seed_product_information=seed_product_info
                )
                prompt = instruction
                
                user_query_obj = await generate_user_query(user_query_runnable, prompt)
                user_query = user_query_obj.model_dump()
                
                return {
                    "user": user,
                    "seed_product_type": item_info[0]['stats']['main_category'],
                    "user_context": user_context,
                    "situation": user_query["situation"],
                    "budget_range": user_query["budget_range"],
                    "user_query": user_query["query"],
                    "seed_product_information": seed_product_info,
                }
        except Exception as e:
            print(f"Error generating user query for user {user}: {e}")
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
        model="gpt-4.1-mini",
        output_version="responses/v1",
        api_key=api_key,
        temperature=1.00,
        max_tokens=2048,
        top_p=1.00,
    )

    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates comprehensive user profiles. Return only valid JSON for the specified schema."),
        ("human", "{input}")
    ])

    def build_runnable():
        return prompt_tmpl | llm.with_structured_output(UserQuery)

    # Resolve domain via mapping; allow full value passthrough if already full
    mapped_domain = CATEGORY_MAPPING.get(args.domain.lower(), args.domain)

    # Load raw data file
    if mapped_domain == 'open_curation':
        raw_data_path = os.path.join(paths["raw_data_dir"], "open_curation.json")
    else:
        raw_data_path = os.path.join(paths["raw_data_dir"], f"{mapped_domain}.json")
    raw_data = json.load(open(raw_data_path, 'r'))
    
    # Load user context data
    user_context_path = os.path.join(paths["user_context_dir"], f"{args.domain}_user_context.json")
    user_context_data = json.load(open(user_context_path, 'r'))
    
    # Convert user context data to dictionary for easy lookup
    user_context_dict = {item["user"]: item for item in user_context_data}
    
    items = list(raw_data.items())
    if args.samples is not None:
        items = items[:max(0, int(args.samples))]
    result = []
    total_len = len(items)

    user_query_runnable = build_runnable()

    # Process users concurrently with rate limiting
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests
    
    # Create tasks for all users
    tasks = [process_user(user_item, user_query_runnable, semaphore, user_context_dict, args.domain) for user_item in items]
    
    # Process with progress bar
    result = []
    with tqdm(total=len(tasks), desc="Generating user queries") as pbar:
        for task in asyncio.as_completed(tasks):
            try:
                user_result = await task
                if user_result is not None:
                    result.append(user_result)
            except Exception as e:
                print(f"Unexpected error: {e}")
            finally:
                pbar.update(1)

    # Create output directory if it doesn't exist
    os.makedirs(paths["output_dir"], exist_ok=True)
    
    output_path = os.path.join(paths["output_dir"], f"{args.domain}_user_query.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    asyncio.run(async_main())
