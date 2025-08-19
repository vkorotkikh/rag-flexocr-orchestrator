"""
RAG FlexOCR Orchestrator - Advanced OCR-focused RAG system.

A comprehensive system for processing unstructured documents with OCR,
intelligent chunking, and retrieval-augmented generation.
"""

from .models import (
    OCRDocument,
    OCRResult,
    OCRChunk,
    ProcessingResult,
    QualityAssessment
)

from .orchestrator import OCRRAGOrchestrator

from .ocr_processors import (
    create_ocr_processor,
    TesseractProcessor,
    EasyOCRProcessor
)

from .chunkers import (
    create_ocr_chunker,
    LayoutAwareChunker,
    ConfidenceBasedChunker,
    HybridSemanticLayoutChunker
)

from .document_analyzers import (
    LayoutAnalyzer,
    StructureDetector
)

from .quality_assessors import (
    OCRQualityAssessor,
    ConfidenceValidator
)

__version__ = "1.0.0"
__author__ = "RAG FlexOCR Team"

__all__ = [
    # Core models
    "OCRDocument",
    "OCRResult", 
    "OCRChunk",
    "ProcessingResult",
    "QualityAssessment",
    
    # Main orchestrator
    "OCRRAGOrchestrator",
    
    # OCR processors
    "create_ocr_processor",
    "TesseractProcessor",
    "EasyOCRProcessor",
    
    # Chunkers
    "create_ocr_chunker",
    "LayoutAwareChunker",
    "ConfidenceBasedChunker", 
    "HybridSemanticLayoutChunker",
    
    # Document analysis
    "LayoutAnalyzer",
    "StructureDetector",
    
    # Quality assessment
    "OCRQualityAssessor",
    "ConfidenceValidator",
]
