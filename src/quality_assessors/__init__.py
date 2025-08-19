"""
Quality assessment module for OCR results evaluation.
"""

from .ocr_quality_assessor import OCRQualityAssessor
from .confidence_validator import ConfidenceValidator

__all__ = [
    "OCRQualityAssessor",
    "ConfidenceValidator"
]
