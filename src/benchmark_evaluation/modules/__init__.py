"""
Modules package for agent-as-judge system.

This package contains the core classes for the product search and evaluation system:
- Extractor: Handles product information extraction from web pages
- Verifier: Compares extracted data against ground truth
- Evaluator: Evaluates product satisfaction against user check_list criteria
"""

from .extractor import Extractor
from .evaluator import Evaluator

__all__ = ['Extractor', 'Evaluator']
