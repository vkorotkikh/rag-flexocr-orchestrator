"""
OCR Chunk models for representing processed text chunks with layout information.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator

from .ocr_result import BoundingBox, RegionType


class ChunkingStrategy(str, Enum):
    """Available chunking strategies for OCR documents."""
    LAYOUT_AWARE = "layout_aware"
    CONFIDENCE_BASED = "confidence_based"
    HYBRID_SEMANTIC_LAYOUT = "hybrid_semantic_layout"
    TABLE_PRESERVING = "table_preserving"
    FIXED_SIZE = "fixed_size"


class ChunkMetadata(BaseModel):
    """Metadata for OCR-generated chunks."""
    # Source information
    document_id: str = Field(..., description="Source document ID")
    page_number: int = Field(..., description="Source page number")
    chunk_index: int = Field(..., description="Index of chunk within document")
    
    # Layout information
    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="Bounding boxes of regions in chunk")
    region_types: List[RegionType] = Field(default_factory=list, description="Types of regions in chunk")
    
    # Quality metrics
    average_confidence: float = Field(..., description="Average OCR confidence for chunk")
    min_confidence: float = Field(..., description="Minimum OCR confidence in chunk")
    max_confidence: float = Field(..., description="Maximum OCR confidence in chunk")
    confidence_variance: float = Field(default=0.0, description="Variance in confidence scores")
    
    # Content characteristics
    word_count: int = Field(default=0, description="Number of words in chunk")
    character_count: int = Field(default=0, description="Number of characters in chunk")
    has_tables: bool = Field(default=False, description="Chunk contains table data")
    has_figures: bool = Field(default=False, description="Chunk contains figure references")
    
    # Structural information
    is_header: bool = Field(default=False, description="Chunk is primarily header content")
    is_footer: bool = Field(default=False, description="Chunk is primarily footer content")
    is_title: bool = Field(default=False, description="Chunk is a title or heading")
    reading_order: Optional[int] = Field(None, description="Reading order within page")
    hierarchy_level: Optional[int] = Field(None, description="Hierarchical level (0=title, 1=section, etc.)")
    
    # Language and formatting
    primary_language: Optional[str] = Field(None, description="Primary language detected")
    contains_multiple_languages: bool = Field(default=False, description="Chunk contains multiple languages")
    font_size_range: Optional[Tuple[float, float]] = Field(None, description="Min and max font sizes")
    has_bold_text: bool = Field(default=False, description="Contains bold formatting")
    has_italic_text: bool = Field(default=False, description="Contains italic formatting")
    
    # Processing metadata
    chunking_strategy: ChunkingStrategy = Field(..., description="Strategy used to create chunk")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Chunk creation timestamp")
    processing_notes: List[str] = Field(default_factory=list, description="Processing notes and warnings")
    
    @validator('average_confidence', 'min_confidence', 'max_confidence')
    def validate_confidence_values(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence values must be between 0.0 and 1.0')
        return v
    
    @property
    def is_high_quality(self) -> bool:
        """Check if chunk is high quality based on confidence."""
        return self.average_confidence >= 0.8 and self.min_confidence >= 0.6
    
    @property
    def is_reliable(self) -> bool:
        """Check if chunk is reliable for RAG usage."""
        return (self.average_confidence >= 0.7 and 
                self.min_confidence >= 0.5 and
                self.word_count >= 3)
    
    @property
    def confidence_range(self) -> float:
        """Calculate confidence range (max - min)."""
        return self.max_confidence - self.min_confidence
    
    def get_spatial_bounds(self) -> Optional[BoundingBox]:
        """Get overall bounding box encompassing all regions."""
        if not self.bounding_boxes:
            return None
        
        min_x1 = min(box.x1 for box in self.bounding_boxes)
        min_y1 = min(box.y1 for box in self.bounding_boxes)
        max_x2 = max(box.x2 for box in self.bounding_boxes)
        max_y2 = max(box.y2 for box in self.bounding_boxes)
        
        return BoundingBox(x1=min_x1, y1=min_y1, x2=max_x2, y2=max_y2)


class OCRChunk(BaseModel):
    """
    A text chunk derived from OCR processing with layout and quality information.
    
    This extends traditional text chunks with OCR-specific metadata including
    confidence scores, spatial information, and layout context.
    """
    # Core content
    content: str = Field(..., description="Text content of the chunk")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    
    # OCR-specific metadata
    metadata: ChunkMetadata = Field(..., description="OCR chunk metadata")
    
    # Vector embedding (populated after embedding generation)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of chunk content")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding generation")
    
    # Relationships to other chunks
    previous_chunk_id: Optional[str] = Field(None, description="ID of previous chunk in sequence")
    next_chunk_id: Optional[str] = Field(None, description="ID of next chunk in sequence")
    related_chunk_ids: List[str] = Field(default_factory=list, description="IDs of semantically related chunks")
    
    # Quality flags
    is_filtered: bool = Field(default=False, description="Chunk was filtered due to quality issues")
    filter_reason: Optional[str] = Field(None, description="Reason for filtering if applicable")
    needs_review: bool = Field(default=False, description="Chunk needs manual review")
    review_notes: List[str] = Field(default_factory=list, description="Manual review notes")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Chunk content cannot be empty')
        return v.strip()
    
    @property
    def clean_content(self) -> str:
        """Get cleaned version of content."""
        import re
        # Remove excessive whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', self.content.strip())
        # Remove problematic characters that might indicate OCR errors
        cleaned = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\"\'\/\\]', '', cleaned)
        return cleaned
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if chunk has high OCR confidence."""
        return self.metadata.average_confidence >= 0.8
    
    @property
    def is_suitable_for_rag(self) -> bool:
        """Check if chunk is suitable for RAG usage."""
        return (not self.is_filtered and 
                self.metadata.is_reliable and
                len(self.clean_content.split()) >= 3)
    
    @property
    def spatial_location(self) -> Optional[BoundingBox]:
        """Get spatial location of chunk."""
        return self.metadata.get_spatial_bounds()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of chunk context for debugging."""
        return {
            "chunk_id": self.chunk_id,
            "page": self.metadata.page_number,
            "word_count": self.metadata.word_count,
            "confidence": f"{self.metadata.average_confidence:.2f}",
            "region_types": list(set(self.metadata.region_types)),
            "has_tables": self.metadata.has_tables,
            "hierarchy_level": self.metadata.hierarchy_level,
            "is_reliable": self.metadata.is_reliable
        }
    
    def to_rag_format(self) -> Dict[str, Any]:
        """Convert chunk to format suitable for RAG vector storage."""
        return {
            "id": self.chunk_id,
            "content": self.clean_content,
            "metadata": {
                "document_id": self.metadata.document_id,
                "page_number": self.metadata.page_number,
                "chunk_index": self.metadata.chunk_index,
                "confidence": self.metadata.average_confidence,
                "word_count": self.metadata.word_count,
                "region_types": [rt.value for rt in self.metadata.region_types],
                "has_tables": self.metadata.has_tables,
                "is_header": self.metadata.is_header,
                "hierarchy_level": self.metadata.hierarchy_level,
                "chunking_strategy": self.metadata.chunking_strategy.value,
                "created_at": self.metadata.created_at.isoformat()
            },
            "embedding": self.embedding
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def model_dump_for_storage(self) -> Dict[str, Any]:
        """Dump model optimized for storage (excluding large fields)."""
        data = self.model_dump()
        # Remove embedding vector for storage efficiency if needed
        if self.embedding and len(self.embedding) > 100:
            data['embedding'] = f"<{len(self.embedding)} dimensions>"
        return data
