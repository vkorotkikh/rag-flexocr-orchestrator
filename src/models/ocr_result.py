"""
OCR Result models for representing extracted text and layout information.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class OCREngine(str, Enum):
    """Supported OCR engines."""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    AWS_TEXTRACT = "aws_textract"
    GOOGLE_VISION = "google_vision"
    AZURE_VISION = "azure_vision"
    TROCR = "trocr"  # Microsoft's Transformer-based OCR
    LAYOUTLMV3 = "layoutlmv3"  # Microsoft's LayoutLMv3 for document understanding
    PADDLEOCR = "paddleocr"  # PaddlePaddle OCR


class RegionType(str, Enum):
    """Types of regions detected in documents."""
    TEXT = "text"
    TITLE = "title"
    HEADER = "header"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTER = "footer"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Bounding box coordinates for text regions."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    
    @validator('x2')
    def x2_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @validator('y2')
    def y2_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v
    
    @property
    def width(self) -> float:
        """Calculate width of bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Calculate height of bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Calculate area of bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def overlaps_with(self, other: "BoundingBox") -> bool:
        """Check if this bounding box overlaps with another."""
        return not (self.x2 < other.x1 or other.x2 < self.x1 or 
                   self.y2 < other.y1 or other.y2 < self.y1)
    
    def intersection_with(self, other: "BoundingBox") -> Optional["BoundingBox"]:
        """Calculate intersection with another bounding box."""
        if not self.overlaps_with(other):
            return None
        
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


class OCRRegion(BaseModel):
    """A region of text detected by OCR with metadata."""
    text: str = Field(..., description="Extracted text content")
    confidence: float = Field(..., description="OCR confidence score (0-1)")
    bounding_box: BoundingBox = Field(..., description="Spatial location of text")
    region_type: RegionType = Field(default=RegionType.TEXT, description="Type of content region")
    language: Optional[str] = Field(None, description="Detected language")
    font_size: Optional[float] = Field(None, description="Estimated font size")
    is_bold: Optional[bool] = Field(None, description="Text appears bold")
    is_italic: Optional[bool] = Field(None, description="Text appears italic")
    reading_order: Optional[int] = Field(None, description="Reading order index")
    
    # Hierarchical structure
    parent_region_id: Optional[str] = Field(None, description="ID of parent region")
    child_region_ids: List[str] = Field(default_factory=list, description="IDs of child regions")
    
    # Internal
    region_id: str = Field(..., description="Unique region identifier")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if region has high confidence (>= 0.8)."""
        return self.confidence >= 0.8
    
    @property
    def is_low_confidence(self) -> bool:
        """Check if region has low confidence (< 0.5)."""
        return self.confidence < 0.5
    
    @property
    def word_count(self) -> int:
        """Count words in the text."""
        return len(self.text.split()) if self.text else 0
    
    def clean_text(self) -> str:
        """Get cleaned version of text."""
        import re
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', self.text.strip())
        return cleaned


class TableCell(BaseModel):
    """A cell within a detected table."""
    text: str = Field(..., description="Cell text content")
    confidence: float = Field(..., description="OCR confidence for this cell")
    row: int = Field(..., description="Row index (0-based)")
    col: int = Field(..., description="Column index (0-based)")
    row_span: int = Field(default=1, description="Number of rows this cell spans")
    col_span: int = Field(default=1, description="Number of columns this cell spans")
    bounding_box: BoundingBox = Field(..., description="Cell bounding box")


class TableStructure(BaseModel):
    """Structure information for detected tables."""
    rows: int = Field(..., description="Number of rows")
    columns: int = Field(..., description="Number of columns")
    cells: List[TableCell] = Field(..., description="Table cells")
    has_header: bool = Field(default=False, description="Table has header row")
    table_type: str = Field(default="grid", description="Type of table structure")
    
    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """Get cell at specific row and column."""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None
    
    def to_text_matrix(self) -> List[List[str]]:
        """Convert table to 2D text matrix."""
        matrix = [["" for _ in range(self.columns)] for _ in range(self.rows)]
        for cell in self.cells:
            if cell.row < self.rows and cell.col < self.columns:
                matrix[cell.row][cell.col] = cell.text
        return matrix


class OCRResult(BaseModel):
    """
    Complete OCR processing result for a document or page.
    """
    document_id: str = Field(..., description="Associated document ID")
    page_number: int = Field(default=1, description="Page number (1-based)")
    
    # OCR extraction results
    regions: List[OCRRegion] = Field(default_factory=list, description="Detected text regions")
    tables: List[TableStructure] = Field(default_factory=list, description="Detected table structures")
    
    # Processing metadata
    ocr_engine: OCREngine = Field(..., description="OCR engine used")
    processing_time: float = Field(..., description="Processing time in seconds")
    image_dimensions: Tuple[int, int] = Field(..., description="Image width and height")
    language_detected: Optional[str] = Field(None, description="Primary detected language")
    
    # Quality metrics
    overall_confidence: float = Field(..., description="Average confidence across all regions")
    text_coverage_ratio: float = Field(..., description="Ratio of image covered by text")
    low_confidence_regions: int = Field(default=0, description="Count of low confidence regions")
    
    # Processing config used
    preprocessing_applied: List[str] = Field(default_factory=list, description="Applied preprocessing steps")
    confidence_threshold: float = Field(default=0.7, description="Confidence threshold used")
    
    # Timing
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    @validator('overall_confidence')
    def validate_overall_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Overall confidence must be between 0.0 and 1.0')
        return v
    
    @validator('text_coverage_ratio')
    def validate_coverage_ratio(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Text coverage ratio must be between 0.0 and 1.0')
        return v
    
    @property
    def total_text(self) -> str:
        """Get all text concatenated in reading order."""
        # Sort by reading order if available, otherwise by vertical then horizontal position
        sorted_regions = sorted(
            self.regions,
            key=lambda r: (
                r.reading_order if r.reading_order is not None else 999,
                r.bounding_box.y1,
                r.bounding_box.x1
            )
        )
        return " ".join(region.text for region in sorted_regions if region.text.strip())
    
    @property
    def high_confidence_text(self) -> str:
        """Get only high-confidence text."""
        high_conf_regions = [r for r in self.regions if r.is_high_confidence]
        sorted_regions = sorted(
            high_conf_regions,
            key=lambda r: (r.bounding_box.y1, r.bounding_box.x1)
        )
        return " ".join(region.text for region in sorted_regions if region.text.strip())
    
    @property
    def is_high_quality(self) -> bool:
        """Check if OCR result is high quality."""
        return (self.overall_confidence >= 0.8 and 
                self.text_coverage_ratio >= 0.1 and
                len(self.regions) > 0)
    
    def get_regions_by_type(self, region_type: RegionType) -> List[OCRRegion]:
        """Get all regions of a specific type."""
        return [r for r in self.regions if r.region_type == region_type]
    
    def get_text_by_confidence(self, min_confidence: float = 0.8) -> str:
        """Get text filtered by minimum confidence."""
        high_conf_regions = [r for r in self.regions if r.confidence >= min_confidence]
        sorted_regions = sorted(
            high_conf_regions,
            key=lambda r: (r.bounding_box.y1, r.bounding_box.x1)
        )
        return " ".join(region.text for region in sorted_regions if region.text.strip())
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
