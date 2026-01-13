"""
Utility functions for product extraction.

Purpose:
    - Contains utility functions used across the extraction system
    - Includes helper functions for DOM manipulation, data processing, and file operations

Usage:
    - Import specific functions as needed: from util import wait_for_render_readiness
    - Used by both extractor.py and extraction.py
"""

import json
from math import e
import re
import base64
import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from PIL import Image

def build_template(obj: Any) -> Any:
    """Build a template structure following item metadata format of web/search agent responses.
    
    Args:
        obj (Any): The object to create a template structure from.
        
    Returns:
        Any: Template with same structure but None values for consistent metadata format.
        Excludes fields: reasoning, user_id, result_index
    """
    if isinstance(obj, dict):
        # Fields to exclude from the template
        excluded_fields = {"reasoning", "user_id", "result_index"}
        return {k: build_template(v) for k, v in obj.items() if k not in excluded_fields}
    if isinstance(obj, list):
        return [build_template(v) for v in obj]
    return None


def load_test_data(file_path: Path) -> Dict[str, Any]:
    """Load test data from JSON file.
    
    Args:
        file_path (Path): Path to the JSON file.
        
    Returns:
        Dict[str, Any]: Loaded test data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_n_samples(test_data: Dict[str, Any], n: int) -> List[Tuple[str, Dict[str, Any]]]:
    """Get N random valid samples from test data.
    
    Args:
        test_data (Dict[str, Any]): Test data dictionary.
        n (int): Number of samples to get.
        
    Returns:
        List[tuple[str, Dict[str, Any]]]: List of randomly selected user ID and user data tuples.
        
    Raises:
        ValueError: If no valid samples found.
    """
    # First, collect all valid samples
    all_valid_samples: List[tuple[str, Dict[str, Any]]] = []
    for user_id, user_data in test_data.items():
        # Check if user_data has the required structure with search_result
        if isinstance(user_data, list) and len(user_data) >= 2 and isinstance(user_data[1], dict) and 'search_result' in user_data[1]:
            # Standard format: user_data is a list with entry[1] containing search_result
            all_valid_samples.append((user_id, user_data[1]))
        elif isinstance(user_data, dict) and 'search_result' in user_data:
            # Flat format: user_data is a dict directly containing search_result
            all_valid_samples.append((user_id, user_data))
    
    if not all_valid_samples:
        raise ValueError("No valid samples found in test data")
    
    # Randomly sample n items (or all if n > available samples)
    sample_size = min(n, len(all_valid_samples))
    return random.sample(all_valid_samples, sample_size)


def save_user_context(user_id: str, user_data: Dict[str, Any], output_dir: Path) -> None:
    """Save user context to user directory.
    
    Args:
        user_id (str): User identifier.
        user_data (Dict[str, Any]): User data containing user_context and check_list.
        output_dir (Path): Output directory for the user.
    """
    # Extract user_context and check_list from the new structure
    user_context = user_data.get('user_context')
    check_list = user_data.get('check_list')
    
    # Create user info with both user_context and check_list
    user_info = {
        'user_context': user_context,
        'check_list': check_list
    }
    
    if user_info:
        user_context_file = output_dir / f"user_{user_id}" / 'user_info.json'
        user_context_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(user_context_file, 'w', encoding='utf-8') as f:
            json.dump(user_info, f, ensure_ascii=False, indent=2)


def convert_to_extractor_input(user_id: str, user_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert test data format to extractor input format for all search results.
    
    Args:
        user_id (str): User identifier.
        user_data (Dict[str, Any]): User data containing search_result array.
        
    Returns:
        List[Dict[str, Any]]: List of data in extractor input format for all search results.
        
    Raises:
        ValueError: If search result format is invalid.
    """
    # Get search_result array directly from user_data
    search_results = user_data.get('search_result')
    
    # Handle different nested structures
    if isinstance(search_results, dict):
        if 'search_result' in search_results and search_results['search_result']:
            search_results = search_results['search_result']
        elif 'search_results' in search_results and search_results['search_results']:
            search_results = search_results['search_results']
        elif 'results' in search_results and search_results['results']:
            search_results = search_results['results']
    elif not isinstance(search_results, list):
        search_results = []

    # Convert all search results
    extractor_inputs = []
    for i, result in enumerate(search_results):
        try:
            extractor_inputs.append({
                "product_title": result['product_title'],
                "product_price": result.get('product_price'),
                "product_url": result['product_url'],
                "product_meta_information": result['product_meta_information'],
                "reasoning": result['reasoning'],
                "user_id": user_id,
                "result_index": i
            })
        except Exception as e:  
            print(type(result))
            print(f"Error storing result {i}: {e}")
    
    return extractor_inputs


def crop_image_into_parts(fullpage_path: Path, artifacts_dir: Path, viewport_height: int) -> List[Path]:
    """Crop fullpage image into adaptive parts based on main page height ratio.
    
    Logic:
    - First capture main page from top
    - If full page height / main page height > 1 and < 2: rest as second section
    - If > 2 and < 3: scroll to capture second part, then rest as 3rd section
    - Continue pattern up to 5 parts maximum
    - If >= 5: crop fullpage into 5 parts
    
    Args:
        fullpage_path (Path): Path to the fullpage screenshot.
        artifacts_dir (Path): Directory to save cropped parts.
        viewport_height (int): Height of the viewport used for main page capture.
        
    Returns:
        List[Path]: List of cropped image paths.
    """
    cropped_parts = []
    
    try:
        with Image.open(fullpage_path) as img:
            width, height = img.size
            
            # Use the actual viewport height used for main page capture
            main_page_height = viewport_height
            height_ratio = height / main_page_height
            
            # Determine number of parts based on height ratio
            if height_ratio <= 1:
                # Single page: no cropping needed
                optimal_parts = 1
            elif height_ratio < 2:
                # 1 < ratio < 2: main page + rest as second section
                optimal_parts = 2
            elif height_ratio < 3:
                # 2 <= ratio < 3: main page + second part + rest as third section
                optimal_parts = 3
            elif height_ratio < 4:
                # 3 <= ratio < 4: main page + second + third + rest as fourth section
                optimal_parts = 4
            else:
                # ratio >= 4: crop into 5 parts maximum
                optimal_parts = 5
            
            # Calculate part heights based on the logic
            if optimal_parts == 1:
                # Single part: use entire image
                cropped_img = img.crop((0, 0, width, height))
                part_path = artifacts_dir / 'fullpage_part_1.png'
                cropped_img.save(part_path)
                cropped_parts.append(part_path)
            else:
                # Multiple parts: first part is main page height, rest divided equally
                first_part_height = main_page_height
                
                # First part: main page from top
                cropped_img = img.crop((0, 0, width, first_part_height))
                part_path = artifacts_dir / 'fullpage_part_1.png'
                cropped_img.save(part_path)
                cropped_parts.append(part_path)
                
                # Remaining parts: divide the rest equally
                remaining_height = height - first_part_height
                remaining_parts = optimal_parts - 1
                part_height = remaining_height // remaining_parts
                
                for i in range(remaining_parts):
                    top = first_part_height + (i * part_height)
                    bottom = height if i == remaining_parts - 1 else top + part_height
                    
                    cropped_img = img.crop((0, top, width, bottom))
                    part_path = artifacts_dir / f'fullpage_part_{i+2}.png'
                    cropped_img.save(part_path)
                    cropped_parts.append(part_path)
                
    except Exception as e:
        print(f"Error cropping fullpage image: {e}")
        return []
    
    return cropped_parts


def encode_image_to_base64(image_path: Path) -> Optional[str]:
    """Encode image file to base64 data URL.
    
    Args:
        image_path (Path): Path to the image file.
        
    Returns:
        Optional[str]: Base64 encoded data URL or None if error.
    """
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode('ascii')
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def prepare_image_messages(cropped_images: List[Path]) -> List[Dict[str, Any]]:
    """Prepare image messages for GPT-5 API calls.
    
    Args:
        cropped_images (List[Path]): List of cropped image paths.
        
    Returns:
        List[Dict[str, Any]]: List of message dictionaries for API calls.
    """
    messages = []
    
    for i, img_path in enumerate(cropped_images):
        try:
            data_url = encode_image_to_base64(img_path)
            if not data_url:
                continue
                
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url
                    }
                ]
            })
            
        except Exception as e:
            print(f"Error preparing image {i+1}: {e}")
            continue
    
    return messages


async def capture_and_crop_fullpage(url: str, artifacts_dir: Path) -> List[Path]:
    """Capture fullpage screenshot and crop into dynamic parts based on viewport height.
    
    Args:
        url (str): URL to capture.
        artifacts_dir (Path): Directory to save images.
        
    Returns:
        List[Path]: List of cropped image paths.
        
    Raises:
        RuntimeError: If screenshot capture or cropping fails.
    """
    
    
    async with Stealth().use_async(async_playwright()) as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale='en-US',
            viewport={'width': 1280, 'height': 720},
            user_agent=random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            ])
        )
        page = await context.new_page()

        try:
            # Navigate to the URL with timeout handling
            await page.goto(url, wait_until='domcontentloaded', timeout=3000)

            # Get actual viewport height from the browser
            viewport_height = await page.evaluate("window.innerHeight")

            # Capture full-page screenshot
            fullpage_path = artifacts_dir / 'fullpage.png'
            await page.screenshot(path=str(fullpage_path), full_page=True)

            # Crop into parts using the actual viewport height
            cropped_parts = crop_image_into_parts(fullpage_path, artifacts_dir, viewport_height)
            return cropped_parts

        except Exception as e:
            # You may want to log e for debugging
            print(f"Error capturing fullpage screenshot: {e}, {url}")
            return []

        finally:
            await context.close()
            await browser.close()