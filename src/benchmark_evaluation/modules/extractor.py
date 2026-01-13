"""
Product Extractor: Structured extraction using OpenAI Responses.parse.

Purpose:
    - Extracts page accessibility, available product info, and reviews from a product URL
    - Uses structured output parsing via Pydantic models
    - Provides an `Extractor` wrapper compatible with the multithreaded pipeline

Usage:
    - Pipeline: extractor = Extractor(client); await extractor.process_item(idx, item, user_out)
"""

from ast import Str
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from utils import (
    capture_and_crop_fullpage,
    prepare_image_messages,
)



class Extractor:
    """Extractor using structured output parsing.

    Exposes `process_item` for compatibility with the multithreaded pipeline.
    """
    
    def __init__(self, client):
        """Initialize the extractor.
        
        Args:
            client: OpenAI client instance.
        """
        self.client = client
        
        
    async def process_item(self, index: int, item: Dict[str, Any], user_out: Path) -> None:
        """Process a single item: call LLM and save structured outputs.

        Args:
            index (int): Item index within the user batch.
            item (Dict[str, Any]): Item dictionary containing at least `product_url` and optional metadata.
            user_out (Path): Base user output directory; results saved under `item_{index:03d}`.
        """
        # Enforce a per-item timeout so a single item cannot block the run
        try:
            await asyncio.wait_for(self._process_item_impl(index, item, user_out), timeout=120)
        except asyncio.TimeoutError:
            # On timeout, write a minimal placeholder so the pipeline can continue
            url = item.get("product_url", "")
            output_dir = user_out
            output_dir.mkdir(parents=True, exist_ok=True)
            await self._save_results(
                output_dir=output_dir,
                extraction_result={
                    "page_accessibility": {"accessible": False, "status": "timeout", "is_specific_product_page": False, "reason": "extraction timed out"},
                    "available_product_info": {"description": ""},
                    "reviews": {"available": False, "details": None},
                },
                url=url,
                item=item,
            )
            return

        return

    async def _process_item_impl(self, index: int, item: Dict[str, Any], user_out: Path) -> None:
        """Internal implementation for processing a single item without timeout wrapper."""
        # Get the URL
        url = item.get("product_url")

        # Prepare directories
        output_dir = user_out
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare artifacts directory for screenshots
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Capture fullpage screenshot and crop into parts
        cropped_images = await capture_and_crop_fullpage(url, artifacts_dir)

        # Extract using structured output parsing, with cropped images and optional URLs
        extraction_result = await self._extract_product_info(url, cropped_images=cropped_images)

        # Save results
        await self._save_results(output_dir=output_dir, extraction_result=extraction_result, url=url, item=item)
        return
            
    
    async def _extract_product_info(self, url: str, cropped_images: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Extract product information using structured output parsing.

        Args:
            url (str): Product URL.

        Returns:
            Dict[str, Any]: Parsed result dict.
        """

        # class PageAccessibility(BaseModel):
        #     accessible: bool
        #     status: str = Field(description="success|error|restricted|not_found|etc")
        #     is_specific_product_page: bool = Field(description="Whether the page is a specific product page")
        #     reason: str = Field(description="Reason for inaccessibility or context")

        class Reviews(BaseModel):
            available: bool
            # Compact JSON as string (not object) to satisfy strict schema
            details: Optional[str] = None
        
        class AvailableProductInfo(BaseModel):
            description: str

        class ExtractionResult(BaseModel):
            # page_accessibility: PageAccessibility  # disabled
            # Compact JSON as string (not object) to satisfy strict schema
            available_product_info: AvailableProductInfo
            reviews: Reviews

        input_prompt = f"""
        Extract structured product information from this URL: {url}
        
        Guidelines:
        1. MUST Navigate to the url and analyze the page content.
        2. You must identify the main product on the page. If the page is not a specific product page, mark as false.
        3. You must extract only information about the main product. Be cautious about distinguishing between the main product and the variants, or advertisements.
        4. Extract all information related to the main product available on the page, including the descriptions, shipping information, etc.
        5. Extract all review information available on the page, including the review summary, raw review texts, review metadata, etc.
        
        I will also provide cropped screenshots of the product page for visual analysis.
        However, you should still use the url as the primary source to extract information from the page content itself through the web search tool.
        The screenshots are supplementary - use them to enhance your extraction but don't rely solely on them.
        """.strip()

        # Build base messages
        messages: List[Dict[str, Any]] = []
        messages.append({
            "role": "system",
            "content": "You are a precise product information extractor."
        })

        # Main prompt as a single user message with input_text
        user_content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": input_prompt}
        ]
        messages.append({"role": "user", "content": user_content})

        # Add cropped images as separate message entries using utils
        if cropped_images:
            image_messages = prepare_image_messages(cropped_images)
            messages.extend(image_messages)

        response = self.client.responses.parse(
            model="gpt-5-mini",
            input=messages,
            text_format=ExtractionResult,
            tools=[{"type": "web_search"}],
            reasoning={"effort": "low"},
        )
        
        parsed: ExtractionResult = response.output_parsed
        result_dict = parsed.model_dump()

        return result_dict
            
    
    async def _save_results(self, output_dir: Path, extraction_result: Dict[str, Any], url: str, item: Dict[str, Any] = None) -> None:
        """Persist extraction results and minimal context to disk.
        
        Args:
            output_dir (Path): Directory where results should be saved.
            extraction_result (Dict[str, Any]): Result object to persist.
            url (str): Source URL.
            item (Dict[str, Any]): Item information.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "extraction_result.json"

        payload = {
            "source_url": url,
            # Disabled: do not persist page accessibility in output
            # "page_accessibility": extraction_result.get("page_accessibility", {}),
            "available_product_info": extraction_result.get("available_product_info", {}),
            "reviews": extraction_result.get("reviews", {}),
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# End of file