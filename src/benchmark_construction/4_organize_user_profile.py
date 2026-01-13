#!/usr/bin/env python3
"""
User Profile Organization:
    Organize generated user profiles by domain and query type.
    
    Usage example:
        python src/benchmark_construction/4_organize_user_profile.py --domain clothing --samples 3
        
    Domains:
        grocery: Grocery and Gourmet Food (creates 3 files)
        clothing: Clothing, Shoes and Jewelry (categorical only)
        electronics: Electronics (categorical only)
        home: Home and Kitchen (categorical only)
        open: Open-ended curation (generic query)
"""
import argparse
import json
import os
from typing import Dict, List, Any

# Domain mapping
CATEGORY_MAPPING: Dict[str, str] = {
    'grocery': 'Grocery_and_Gourmet_Food',
    'clothing': 'Clothing_Shoes_and_Jewelry',
    'electronics': 'Electronics',
    'home': 'Home_and_Kitchen',
    'open': 'open_curation',
}

# Query type mapping for different domains
QUERY_TYPE_MAPPING = {
    'clothing': 'categorical',
    'electronics': 'categorical', 
    'home': 'categorical',
    'grocery': 'target_finding',  # Will create three separate files
    'open': 'generic'             # Will use generic query
}

def get_project_paths():
    """Get project root and other important paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    return {
        "project_root": project_root,
        "input_dir": os.path.join(project_root, "data", "_user_data", "_user_checklist"),
        "output_dir": os.path.join(project_root, "data", "user_profiles")
    }

def organize_user_profile(user_data: Dict[str, Any], query_type: str) -> Dict[str, Any]:
    """Organize a single user profile based on query type.
    
    Args:
        user_data: Original user data from checklist file
        query_type: Type of query to identify ('categorical', 'explicit_title', 'attribute_specific', 'brand_categorical', 'generic')
        
    Returns:
        Organized user profile data
    """
    organized_profile = {
        "user": user_data["user"],
        "seed_product_type": user_data["seed_product_type"],
        "situation": user_data["situation"],
        "budget_range": user_data["budget_range"],
        "query_type": query_type,
        "user_query": user_data["user_query"],
        "user_context": user_data["user_context"],
        "check_list": user_data["check_list"]
    }
    
    return organized_profile

def process_domain(domain: str, samples: int = None):
    """Process user profiles for a specific domain.
    
    Args:
        domain: Domain to process
        samples: Number of samples to process (None for all)
    """
    paths = get_project_paths()
    query_type = QUERY_TYPE_MAPPING.get(domain)
    
    # Create output directory
    os.makedirs(paths["output_dir"], exist_ok=True)
    
    if query_type == 'categorical' or query_type == 'generic':
        # For clothing, electronics, home, open - single file
        input_file = os.path.join(paths["input_dir"], f"{domain}_user_checklist.json")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            checklist_data = json.load(f)
        
        # Limit samples if specified
        if samples is not None:
            checklist_data = checklist_data[:samples]
        
        organized_data = []
        for user_data in checklist_data:
            organized_profile = organize_user_profile(user_data, query_type)
            organized_data.append(organized_profile)
        
        # Save single file
        output_file = os.path.join(paths["output_dir"], f"{domain}_user_profile.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, indent=4)
        
    elif query_type == 'target_finding':
        # For grocery - read three separate files and copy them
        query_variations = ['explicit_title', 'attribute_specific', 'brand_categorical']
        
        for variation in query_variations:
            input_file = os.path.join(paths["input_dir"], f"{domain}_{variation}_user_checklist.json")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                checklist_data = json.load(f)
            
            # Limit samples if specified
            if samples is not None:
                checklist_data = checklist_data[:samples]
            
            organized_data = []
            for user_data in checklist_data:
                organized_profile = organize_user_profile(user_data, variation)
                organized_data.append(organized_profile)
            
            # Save variation file
            output_file = os.path.join(paths["output_dir"], f"{domain}_{variation}_user_profile.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(organized_data, f, indent=4)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="clothing", 
                       help="Domain short key, e.g., clothing, grocery, electronics, home, open")
    parser.add_argument("--samples", type=int, default=None, 
                       help="Number of samples to process (default: all)")
    return parser.parse_args()

def main():
    args = argparser()
    
    # Validate domain
    if args.domain not in CATEGORY_MAPPING:
        print(f"Error: Invalid domain '{args.domain}'. Valid domains: {list(CATEGORY_MAPPING.keys())}")
        return
    
    process_domain(args.domain, args.samples)

if __name__ == "__main__":
    main()
