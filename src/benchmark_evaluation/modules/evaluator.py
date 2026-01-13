#!/usr/bin/env python3
"""
Check_list Evaluator:
    Evaluate verified product information against user checklist criteria using OpenAI.

Purpose: 
    Analyze verification_result.json files to determine how well each product satisfies
    user-specific checklist requirements. Provides binary evaluation (satisfied/not satisfied)
    for each checklist item with detailed annotations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

EVALUATION_SYSTEM_PROMPT = """
    You are an expert product evaluation assistant. Your task is to evaluate how well a model response (product information) satisfies user-specific checklist criteria.

    You will receive:
    1. MODEL_RESPONSE (product data)
    2. CHECKLIST (product evaluation criteria)

    Your job:
    1. For each item in the checklist, determine if it is SUCCESS or FAIL based on the product data
    2. Provide clear reasoning for each evaluation
    3. Count the total SUCCESS vs FAIL items
    4. Calculate the SUCCESS percentage
    
    Guidelines:
    1. If the product data is not available, annotate FAIL.
    2. If the url is valid specific product page, search the web through the provided url in MODEL_RESPONSE to get more information.
    3. Provide grounded reasoning and evaluation for each checklist item.

    Return ONLY valid JSON with this exact structure:
    {{
        "stats": {{
            "is_product_page": boolean,
            "total_checklist_items": 0,
            "satisfaction_percentage": 0.0,
            "satisfied_items": 0,
            "not_satisfied_items": 0,
            "brand_preferences_percentage": "num_satisfied / num_total",
            "price_sensitivity_percentage": "num_satisfied / num_total",
            "review_sensitivity_percentage": "num_satisfied / num_total",
            "functional_requirements_percentage": "num_satisfied / num_total",
            "aesthetic_preferences_percentage": "num_satisfied / num_total",
            "purchase_preferences_percentage": "num_satisfied / num_total"
        }},
        "annotated_checklist": {{
            "brand_preferences": [
                {{
                    "context": "original statement of the first requirement",
                    "status": "SUCCESS",
                    "reasoning": "reasoning."
                }},
                {{
                    "context": "original statement of the second requirement",
                    "status": "FAIL",
                    "reasoning": "reasoning."
                }}
            ],
            "price_sensitivity": [],
            "review_sensitivity": [],
            "functional_requirements": [],
            "aesthetic_preferences": [],
            "purchase_preferences": []
        }}
    }}
"""

class PageAccessibility(BaseModel):
    accessible: bool
    status: str = Field(description="success|error|restricted|not_found|etc")
    is_specific_product_page: bool = Field(description="Whether the page is a specific product page")
    reason: str = Field(description="Reason for inaccessibility or context")


class Stats(BaseModel):
    is_product_page: bool
    total_checklist_items: int
    satisfaction_percentage: float
    satisfied_items: int
    not_satisfied_items: int
    brand_preferences_percentage: str
    price_sensitivity_percentage: str
    review_sensitivity_percentage: str
    functional_requirements_percentage: str
    aesthetic_preferences_percentage: str
    purchase_preferences_percentage: str

class ChecklistItem(BaseModel):
    context: str
    status: str
    reasoning: str

class AnnotatedChecklist(BaseModel):
    brand_preferences: list[ChecklistItem]
    price_sensitivity: list[ChecklistItem]
    review_sensitivity: list[ChecklistItem]
    functional_requirements: list[ChecklistItem]
    aesthetic_preferences: list[ChecklistItem]
    purchase_preferences: list[ChecklistItem]

class EvaluationOutput(BaseModel):
    page_accessibility: PageAccessibility
    stats: Stats
    annotated_checklist: AnnotatedChecklist


class Evaluator:
    """Evaluator class for assessing product satisfaction against user checklist criteria."""
    
    def __init__(self, client: OpenAI):
        """Initialize the evaluator with an OpenAI client.
        
        Args:
            client (OpenAI): OpenAI client instance for API calls.
        """
        self.client = client
        self._lock = threading.Lock()
    
    def evaluate_product(self, model_data: Dict[str, Any], checklist: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate how well a model response satisfies user checklist criteria.
        
        Args:
            model_data: Dictionary containing model response under 'search_output'
            checklist: Dictionary with user checklist requirements
            
        Returns:
            Evaluation report with satisfaction statistics and annotated checklist
        """
        model_response = model_data.get('search_output', {})
        
        # Use OpenAI to evaluate product against checklist
        evaluation_result = self._call_openai_evaluation(model_response, checklist)
        
        return evaluation_result
    
    def _call_openai_evaluation(self, model_response: Dict[str, Any], checklist: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI responses.parse to evaluate product satisfaction against checklist."""

        user_prompt = f"""
        MODEL_RESPONSE (product data):
        {json.dumps(model_response, ensure_ascii=False, indent=2)}

        CHECK_LIST (user requirements):
        {json.dumps(checklist, ensure_ascii=False, indent=2)}
        """.strip()

        messages = [
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
        ]

        with self._lock:
            response = self.client.responses.parse(
                model="gpt-5-mini",
                input=messages,
                text_format=EvaluationOutput,
                tools=[{"type": "web_search"}],
                reasoning={"effort": "low"},
            )

        parsed: EvaluationOutput = response.output_parsed
        return parsed.model_dump()
    
    def evaluate_batch(self, evaluation_tasks: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]], 
                      max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple verification results in parallel using multithreading.
        
        Args:
            evaluation_tasks: List of tuples containing (file_path, verification_data, checklist)
            max_workers: Maximum number of worker threads (defaults to number of tasks)
            
        Returns:
            List of evaluation reports
        """
        max_workers = max_workers or len(evaluation_tasks)
        all_reports = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks
            future_to_task = {}
            for file_path, verification_data, checklist in evaluation_tasks:
                future = executor.submit(self.evaluate_product, verification_data, checklist)
                future_to_task[future] = file_path
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                file_path = future_to_task[future]
                report = future.result()
                all_reports.append(report)
        
        return all_reports
