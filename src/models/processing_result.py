"""
Processing result models for tracking OCR-RAG pipeline execution.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field, validator

from .ocr_chunk import OCRChunk
from .ocr_result import OCRResult


class ProcessingStatus(str, Enum):
    """Status of document processing."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    OCR_EXTRACTION = "ocr_extraction"
    QUALITY_ASSESSMENT = "quality_assessment"
    LAYOUT_ANALYSIS = "layout_analysis"
    CHUNKING = "chunking"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_INDEXING = "vector_indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingError(BaseModel):
    """Error information for failed processing steps."""
    step: str = Field(..., description="Processing step where error occurred")
    error_type: str = Field(..., description="Type/class of error")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When error occurred")
    is_recoverable: bool = Field(default=True, description="Whether error is recoverable")
    retry_count: int = Field(default=0, description="Number of retry attempts")


class ProcessingMetrics(BaseModel):
    """Performance and quality metrics for processing."""
    # Timing metrics
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    preprocessing_time: float = Field(default=0.0, description="Image preprocessing time")
    ocr_time: float = Field(default=0.0, description="OCR extraction time")
    chunking_time: float = Field(default=0.0, description="Chunking time")
    embedding_time: float = Field(default=0.0, description="Embedding generation time")
    indexing_time: float = Field(default=0.0, description="Vector indexing time")
    
    # Quality metrics
    pages_processed: int = Field(default=0, description="Number of pages processed")
    pages_failed: int = Field(default=0, description="Number of pages that failed")
    total_chunks_created: int = Field(default=0, description="Total chunks created")
    high_quality_chunks: int = Field(default=0, description="High quality chunks")
    filtered_chunks: int = Field(default=0, description="Chunks filtered due to quality")
    
    # OCR quality metrics
    average_ocr_confidence: float = Field(default=0.0, description="Average OCR confidence across document")
    min_page_confidence: float = Field(default=0.0, description="Minimum page confidence")
    max_page_confidence: float = Field(default=0.0, description="Maximum page confidence")
    confidence_variance: float = Field(default=0.0, description="Variance in page confidences")
    
    # Content metrics
    total_text_extracted: int = Field(default=0, description="Total characters extracted")
    tables_detected: int = Field(default=0, description="Number of tables detected")
    figures_detected: int = Field(default=0, description="Number of figures detected")
    languages_detected: List[str] = Field(default_factory=list, description="Languages detected")
    
    # Resource usage
    memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage in MB")
    cpu_time_seconds: Optional[float] = Field(None, description="CPU time used")
    
    @property
    def processing_speed_pages_per_minute(self) -> float:
        """Calculate processing speed in pages per minute."""
        if self.total_processing_time > 0:
            return (self.pages_processed / self.total_processing_time) * 60
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of page processing."""
        total_pages = self.pages_processed + self.pages_failed
        if total_pages > 0:
            return self.pages_processed / total_pages
        return 0.0
    
    @property
    def quality_rate(self) -> float:
        """Calculate rate of high-quality chunks."""
        if self.total_chunks_created > 0:
            return self.high_quality_chunks / self.total_chunks_created
        return 0.0
    
    @property
    def is_high_quality_result(self) -> bool:
        """Check if processing result is high quality."""
        return (self.average_ocr_confidence >= 0.8 and
                self.success_rate >= 0.9 and
                self.quality_rate >= 0.7)


class ProcessingResult(BaseModel):
    """
    Complete result of OCR-RAG pipeline processing.
    
    Contains all extracted data, quality metrics, and processing information.
    """
    # Identification
    document_id: str = Field(..., description="Processed document ID")
    session_id: Optional[str] = Field(None, description="Processing session ID")
    
    # Status and timing
    status: ProcessingStatus = Field(..., description="Final processing status")
    started_at: datetime = Field(..., description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    
    # Processed data
    ocr_results: List[OCRResult] = Field(default_factory=list, description="OCR results per page")
    chunks: List[OCRChunk] = Field(default_factory=list, description="Generated text chunks")
    
    # Processing information
    metrics: ProcessingMetrics = Field(..., description="Processing performance metrics")
    errors: List[ProcessingError] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    
    # Configuration used
    processing_config: Dict[str, Any] = Field(default_factory=dict, description="Configuration used for processing")
    ocr_engines_used: List[str] = Field(default_factory=list, description="OCR engines used")
    chunking_strategy: Optional[str] = Field(None, description="Chunking strategy used")
    
    # Quality assessment
    overall_quality_score: float = Field(default=0.0, description="Overall quality score (0-1)")
    quality_assessment: Optional[Dict[str, Any]] = Field(None, description="Detailed quality assessment")
    
    @validator('overall_quality_score')
    def validate_quality_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Quality score must be between 0.0 and 1.0')
        return v
    
    @property
    def processing_duration(self) -> Optional[timedelta]:
        """Calculate total processing duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_successful(self) -> bool:
        """Check if processing completed successfully."""
        return self.status == ProcessingStatus.COMPLETED and len(self.chunks) > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if processing had errors."""
        return len(self.errors) > 0
    
    @property
    def rag_ready_chunks(self) -> List[OCRChunk]:
        """Get chunks that are suitable for RAG usage."""
        return [chunk for chunk in self.chunks if chunk.is_suitable_for_rag]
    
    @property
    def high_confidence_chunks(self) -> List[OCRChunk]:
        """Get only high-confidence chunks."""
        return [chunk for chunk in self.chunks if chunk.is_high_confidence]
    
    def get_chunks_by_page(self, page_number: int) -> List[OCRChunk]:
        """Get all chunks from a specific page."""
        return [chunk for chunk in self.chunks 
                if chunk.metadata.page_number == page_number]
    
    def get_text_by_confidence(self, min_confidence: float = 0.8) -> str:
        """Get all text with minimum confidence threshold."""
        high_conf_chunks = [chunk for chunk in self.chunks 
                           if chunk.metadata.average_confidence >= min_confidence]
        return " ".join(chunk.clean_content for chunk in high_conf_chunks)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the processing result."""
        rag_ready = len(self.rag_ready_chunks)
        high_conf = len(self.high_confidence_chunks)
        
        return {
            "document_id": self.document_id,
            "status": self.status.value,
            "pages_processed": self.metrics.pages_processed,
            "total_chunks": len(self.chunks),
            "rag_ready_chunks": rag_ready,
            "high_confidence_chunks": high_conf,
            "processing_time": f"{self.metrics.total_processing_time:.2f}s",
            "average_confidence": f"{self.metrics.average_ocr_confidence:.2f}",
            "overall_quality": f"{self.overall_quality_score:.2f}",
            "success_rate": f"{self.metrics.success_rate:.2f}",
            "has_errors": self.has_errors,
            "tables_detected": self.metrics.tables_detected,
            "languages": self.metrics.languages_detected
        }
    
    def add_error(self, step: str, error: Exception, is_recoverable: bool = True):
        """Add an error to the processing result."""
        processing_error = ProcessingError(
            step=step,
            error_type=type(error).__name__,
            message=str(error),
            is_recoverable=is_recoverable
        )
        self.errors.append(processing_error)
        
        if not is_recoverable:
            self.status = ProcessingStatus.FAILED
    
    def add_warning(self, message: str):
        """Add a warning to the processing result."""
        timestamp = datetime.utcnow().isoformat()
        self.warnings.append(f"[{timestamp}] {message}")
    
    def finalize(self):
        """Finalize the processing result."""
        if self.status not in [ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]:
            self.status = ProcessingStatus.COMPLETED
        
        self.completed_at = datetime.utcnow()
        
        # Update metrics
        if self.completed_at and self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            self.metrics.total_processing_time = duration
        
        # Calculate overall quality score
        if self.chunks:
            confidence_scores = [chunk.metadata.average_confidence for chunk in self.chunks]
            self.metrics.average_ocr_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Quality score based on confidence, success rate, and chunk quality
            quality_factors = [
                self.metrics.average_ocr_confidence,
                self.metrics.success_rate,
                self.metrics.quality_rate
            ]
            self.overall_quality_score = sum(quality_factors) / len(quality_factors)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            timedelta: lambda v: v.total_seconds()
        }
