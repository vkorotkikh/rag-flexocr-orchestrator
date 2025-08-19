"""
OCR Document models for representing input documents and their metadata.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Supported document types for OCR processing."""
    PDF = "pdf"
    IMAGE = "image"
    SCANNED_PDF = "scanned_pdf"
    MULTI_PAGE_TIFF = "multi_page_tiff"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata for OCR documents."""
    source: str = Field(..., description="Source file path or identifier")
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    creation_date: Optional[datetime] = Field(None, description="Document creation date")
    page_count: Optional[int] = Field(None, description="Number of pages in document")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    language: Optional[str] = Field(None, description="Primary language of document")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessingConfig(BaseModel):
    """Configuration for OCR processing."""
    enhance_quality: bool = Field(default=True, description="Enable image quality enhancement")
    preserve_layout: bool = Field(default=True, description="Preserve document layout structure")
    detect_language: bool = Field(default=True, description="Auto-detect document language")
    confidence_threshold: float = Field(default=0.7, description="Minimum OCR confidence threshold")
    ocr_engines: List[str] = Field(default=["easyocr"], description="OCR engines to use in order")
    preprocessing_steps: List[str] = Field(
        default=["noise_reduction", "contrast_enhancement", "deskew"],
        description="Image preprocessing steps"
    )
    max_resolution: int = Field(default=2400, description="Maximum image resolution for processing")
    enable_table_detection: bool = Field(default=True, description="Enable table structure detection")
    enable_figure_detection: bool = Field(default=True, description="Enable figure/image detection")
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v
    
    @validator('ocr_engines')
    def validate_engines(cls, v):
        supported_engines = ["tesseract", "easyocr", "aws_textract", "google_vision", "azure_vision"]
        for engine in v:
            if engine not in supported_engines:
                raise ValueError(f'Unsupported OCR engine: {engine}. Supported: {supported_engines}')
        return v


class OCRDocument(BaseModel):
    """
    Represents a document to be processed with OCR.
    
    This is the primary input model for the OCR-RAG system.
    """
    file_path: Optional[str] = Field(None, description="Path to the document file")
    content: Optional[bytes] = Field(None, description="Raw document content as bytes")
    document_type: DocumentType = Field(default=DocumentType.UNKNOWN, description="Type of document")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig, description="OCR processing configuration")
    
    # Internal tracking
    id: Optional[str] = Field(None, description="Unique document identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Document creation timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f'File does not exist: {v}')
            if not path.is_file():
                raise ValueError(f'Path is not a file: {v}')
        return v
    
    @validator('document_type', pre=True, always=True)
    def auto_detect_document_type(cls, v, values):
        """Auto-detect document type from file extension if not specified."""
        if v == DocumentType.UNKNOWN and 'file_path' in values and values['file_path']:
            file_path = Path(values['file_path'])
            extension = file_path.suffix.lower()
            
            if extension == '.pdf':
                v = DocumentType.PDF
            elif extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                if extension in ['.tiff', '.tif']:
                    v = DocumentType.MULTI_PAGE_TIFF
                else:
                    v = DocumentType.IMAGE
        return v
    
    def update_metadata_from_file(self):
        """Update metadata fields from file system information."""
        if self.file_path:
            path = Path(self.file_path)
            if path.exists():
                stat = path.stat()
                self.metadata.file_size = stat.st_size
                self.metadata.creation_date = datetime.fromtimestamp(stat.st_ctime)
                if not self.metadata.title:
                    self.metadata.title = path.stem
                if not self.metadata.source:
                    self.metadata.source = str(path)
    
    @property
    def is_image_document(self) -> bool:
        """Check if document is an image type."""
        return self.document_type in [DocumentType.IMAGE, DocumentType.MULTI_PAGE_TIFF]
    
    @property
    def is_pdf_document(self) -> bool:
        """Check if document is a PDF type."""
        return self.document_type in [DocumentType.PDF, DocumentType.SCANNED_PDF]
    
    @property
    def requires_ocr(self) -> bool:
        """Check if document requires OCR processing."""
        return self.document_type in [
            DocumentType.IMAGE, 
            DocumentType.SCANNED_PDF, 
            DocumentType.MULTI_PAGE_TIFF
        ]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            bytes: lambda v: v.hex() if v else None
        }
        
    def model_dump_safe(self) -> Dict[str, Any]:
        """Dump model data without potentially large binary content."""
        data = self.model_dump()
        if 'content' in data and data['content']:
            data['content'] = f"<{len(self.content)} bytes>" if self.content else None
        return data
