"""
Data models for OCR-focused RAG system.
"""

from .ocr_document import OCRDocument, DocumentMetadata
from .ocr_result import OCRResult, OCRRegion, BoundingBox
from .ocr_chunk import OCRChunk, ChunkMetadata
from .processing_result import ProcessingResult, ProcessingMetrics
from .quality_assessment import QualityAssessment, ConfidenceMetrics

__all__ = [
    "OCRDocument",
    "DocumentMetadata", 
    "OCRResult",
    "OCRRegion",
    "BoundingBox",
    "OCRChunk",
    "ChunkMetadata",
    "ProcessingResult",
    "ProcessingMetrics",
    "QualityAssessment",
    "ConfidenceMetrics"
]

