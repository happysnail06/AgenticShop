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
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
import extruct
from pydantic import BaseModel, Field
from utils import (
    capture_and_crop_fullpage,
    prepare_image_messages,
)
from playwright.async_api import async_playwright
from playwright_stealth import Stealth



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
        extraction_result = await self._extract_product_info_(url, cropped_images=cropped_images)

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
            
    
    async def _fetch_and_parse_page(self, url: str) -> Dict[str, Any]:
        """Fetch page HTML via Playwright and parse with BeautifulSoup + extruct.

        Args:
            url (str): Product URL to fetch.

        Returns:
            Dict[str, Any]: {"clean_text": str, "structured_data": dict} where
                structured_data contains JSON-LD, microdata, and OpenGraph metadata.
        """
        result = {"clean_text": "", "structured_data": {}}

        try:
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
                    await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                    await page.wait_for_timeout(2000)
                    html = await page.content()
                finally:
                    await context.close()
                    await browser.close()

            # Parse clean text with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav']):
                tag.decompose()
            clean_text = soup.get_text(separator='\n', strip=True)
            lines = [line for line in clean_text.splitlines() if line.strip()]
            result["clean_text"] = '\n'.join(lines)

            # Extract structured data with extruct
            metadata = extruct.extract(html, base_url=url, syntaxes=['json-ld', 'microdata', 'opengraph'])
            result["structured_data"] = {k: v for k, v in metadata.items() if v}

        except Exception as e:
            print(f"Error fetching/parsing page {url}: {e}")

        return result

    async def _extract_product_info_(self, url: str, cropped_images: Optional[List[Path]] = None) -> Dict[str, Any]:
        """Product extraction

        Args:
            url (str): Product URL.
            cropped_images (Optional[List[Path]]): Cropped screenshot paths.

        Returns:
            Dict[str, Any]: Parsed result dict matching ExtractionResult schema.
        """
        # Fetch and parse page content
        page_content = await self._fetch_and_parse_page(url)
        clean_text = page_content.get("clean_text", "")
        structured_data = page_content.get("structured_data", {})
        structured_str = json.dumps(structured_data, ensure_ascii=False, indent=2) if structured_data else ""

        class Reviews(BaseModel):
            available: bool
            details: Optional[str] = None

        class AvailableProductInfo(BaseModel):
            description: str

        class ExtractionResult(BaseModel):
            available_product_info: AvailableProductInfo
            reviews: Reviews

        # Build prompt with all available context
        content_sections = [f"Extract structured product information from this URL: {url}"]

        if clean_text:
            content_sections.append(
                f"--- Parsed Page Content ---\n{clean_text}"
            )

        if structured_str:
            content_sections.append(
                f"--- Structured Metadata (JSON-LD / Microdata / OpenGraph) ---\n{structured_str}"
            )

        content_sections.append("""Guidelines:
1. Use the parsed page content and structured metadata as the primary sources for extraction.
2. Use the web search tool to verify or supplement information if the parsed content is incomplete.
3. Identify the main product on the page. Be cautious about distinguishing between the main product and variants or advertisements.
4. Extract all information related to the main product, including descriptions, pricing, specifications, shipping information, etc.
5. Extract all review information available, including review summary, raw review texts, and review metadata.
6. Cropped screenshots of the page are provided for visual verification — use them to cross-check extracted information.""")

        input_prompt = '\n\n'.join(content_sections)

        # Build messages
        messages: List[Dict[str, Any]] = []
        messages.append({
            "role": "system",
            "content": "You are a precise product information extractor."
        })

        user_content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": input_prompt}
        ]
        messages.append({"role": "user", "content": user_content})

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