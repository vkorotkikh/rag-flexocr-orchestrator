"""
Tesseract OCR processor implementation.
"""

import cv2
import numpy as np
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging

from .base import BaseOCRProcessor
from ..models import OCRResult, OCRRegion, BoundingBox, RegionType, OCREngine

logger = logging.getLogger(__name__)


class TesseractProcessor(BaseOCRProcessor):
    """Tesseract OCR processor with advanced configuration options."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 enable_preprocessing: bool = True,
                 preprocessing_steps: Optional[List[str]] = None,
                 language: str = 'eng',
                 page_segmentation_mode: int = 6,
                 ocr_engine_mode: int = 3):
        """
        Initialize Tesseract processor.
        
        Args:
            confidence_threshold: Minimum confidence for text acceptance
            enable_preprocessing: Whether to apply image preprocessing
            preprocessing_steps: List of preprocessing steps
            language: Tesseract language code (e.g., 'eng', 'fra', 'deu')
            page_segmentation_mode: PSM mode (1-13)
            ocr_engine_mode: OEM mode (0-3)
        """
        super().__init__(
            engine_name='tesseract',
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing,
            preprocessing_steps=preprocessing_steps
        )
        
        self.language = language
        self.psm = page_segmentation_mode
        self.oem = ocr_engine_mode
        
        # Tesseract configuration
        self.config = f'--oem {self.oem} --psm {self.psm} -l {self.language}'
        
        # Initialize Tesseract
        self._initialize_tesseract()
    
    def _initialize_tesseract(self):
        """Initialize and validate Tesseract installation."""
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
            # Test Tesseract availability
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
            
            # Get available languages
            available_langs = pytesseract.get_languages()
            if self.language not in available_langs:
                self.logger.warning(f"Language {self.language} not available. Available: {available_langs}")
                self.language = 'eng'  # Fallback to English
            
            self.is_initialized = True
            
        except ImportError:
            self.logger.error("pytesseract not installed. Install with: pip install pytesseract")
            self.is_initialized = False
        except Exception as e:
            self.logger.error(f"Failed to initialize Tesseract: {e}")
            self.is_initialized = False
    
    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        return self.is_initialized
    
    async def extract_text_from_image(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image: Preprocessed image as numpy array
            **kwargs: Additional parameters
            
        Returns:
            OCR result with extracted text and regions
        """
        if not self.is_available():
            raise RuntimeError("Tesseract OCR is not available")
        
        start_time = datetime.now()
        
        document_id = kwargs.get('document_id', str(uuid.uuid4()))
        page_number = kwargs.get('page_number', 1)
        detect_tables = kwargs.get('detect_tables', True)
        detect_figures = kwargs.get('detect_figures', True)
        
        try:
            # Get detailed OCR data
            ocr_data = self.pytesseract.image_to_data(
                image, 
                config=self.config,
                output_type=self.pytesseract.Output.DICT
            )
            
            # Extract regions from OCR data
            regions = self._extract_regions_from_data(ocr_data, image.shape)
            
            # Detect tables if enabled
            tables = []
            if detect_tables:
                tables = self._detect_tables(image, regions)
            
            # Calculate metrics
            overall_confidence = self._calculate_overall_confidence(regions)
            text_coverage = self._calculate_text_coverage(regions, image.shape[:2][::-1])
            low_conf_count = len([r for r in regions if r.confidence < 0.5])
            
            # Detect primary language
            full_text = ' '.join([r.text for r in regions if r.text.strip()])
            detected_language = self._detect_language(full_text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                document_id=document_id,
                page_number=page_number,
                regions=regions,
                tables=tables,
                ocr_engine=OCREngine.TESSERACT,
                processing_time=processing_time,
                image_dimensions=(image.shape[1], image.shape[0]),
                language_detected=detected_language,
                overall_confidence=overall_confidence,
                text_coverage_ratio=text_coverage,
                low_confidence_regions=low_conf_count,
                preprocessing_applied=self.preprocessing_steps,
                confidence_threshold=self.confidence_threshold
            )
            
        except Exception as e:
            self.logger.error(f"Tesseract OCR failed: {e}")
            raise
    
    def _extract_regions_from_data(self, ocr_data: Dict, image_shape: Tuple) -> List[OCRRegion]:
        """Extract text regions from Tesseract OCR data."""
        regions = []
        
        # Group data by text blocks
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = float(ocr_data['conf'][i])
            
            # Skip empty text or very low confidence
            if not text or conf < 0:
                continue
            
            # Extract bounding box
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            if w <= 0 or h <= 0:
                continue
            
            bounding_box = BoundingBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h)
            )
            
            # Determine region type based on properties
            region_type = self._determine_region_type(
                text, 
                conf, 
                bounding_box, 
                image_shape
            )
            
            # Create region
            region = OCRRegion(
                text=text,
                confidence=conf / 100.0,  # Convert to 0-1 range
                bounding_box=bounding_box,
                region_type=region_type,
                region_id=str(uuid.uuid4())
            )
            
            regions.append(region)
        
        # Sort regions by reading order (top to bottom, left to right)
        regions.sort(key=lambda r: (r.bounding_box.y1, r.bounding_box.x1))
        
        # Assign reading order
        for idx, region in enumerate(regions):
            region.reading_order = idx
        
        return regions
    
    def _determine_region_type(self, text: str, confidence: float, 
                              bbox: BoundingBox, image_shape: Tuple) -> RegionType:
        """Determine the type of text region based on characteristics."""
        text_lower = text.lower().strip()
        
        # Header detection (top of page, larger text, short lines)
        if bbox.y1 < image_shape[0] * 0.15 and len(text) < 100:
            return RegionType.HEADER
        
        # Footer detection (bottom of page)
        if bbox.y1 > image_shape[0] * 0.85:
            return RegionType.FOOTER
        
        # Title detection (shorter text, likely centered or large)
        if len(text) < 80 and bbox.height > 20:
            return RegionType.TITLE
        
        # List detection (starts with bullet points or numbers)
        if (text_lower.startswith(('•', '·', '-', '*')) or 
            (len(text) > 2 and text[0].isdigit() and text[1] in '.):')):
            return RegionType.LIST
        
        # Caption detection (starts with "Figure", "Table", etc.)
        caption_keywords = ['figure', 'table', 'chart', 'graph', 'image', 'fig.', 'tab.']
        if any(text_lower.startswith(keyword) for keyword in caption_keywords):
            return RegionType.CAPTION
        
        # Default to paragraph for normal text
        return RegionType.PARAGRAPH
    
    def _detect_tables(self, image: np.ndarray, regions: List[OCRRegion]) -> List:
        """Detect table structures in the image."""
        # This is a simplified table detection
        # In production, you might use more sophisticated methods
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
            
            # Detect vertical lines
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
            
            # Combine lines
            table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find table contours
            contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                # Filter by size to avoid noise
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum table size
                    x, y, w, h = cv2.boundingRect(contour)
                    # This would be expanded to create proper TableStructure objects
                    # For now, return empty list
                    pass
            
            return tables
            
        except Exception as e:
            self.logger.warning(f"Table detection failed: {e}")
            return []
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of extracted text."""
        try:
            from langdetect import detect
            if text and len(text.strip()) > 20:
                return detect(text)
        except:
            pass
        return self.language
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        if self.is_available():
            try:
                return self.pytesseract.get_languages()
            except:
                pass
        return ['eng']  # Default fallback
    
    def set_language(self, language: str):
        """Set OCR language."""
        if language in self.get_supported_languages():
            self.language = language
            self.config = f'--oem {self.oem} --psm {self.psm} -l {self.language}'
        else:
            self.logger.warning(f"Language {language} not supported")
    
    def set_page_segmentation_mode(self, psm: int):
        """
        Set page segmentation mode.
        
        PSM modes:
        0: Orientation and script detection (OSD) only
        1: Automatic page segmentation with OSD
        2: Automatic page segmentation, but no OSD, or OCR
        3: Fully automatic page segmentation, but no OSD
        4: Assume a single column of text of variable sizes
        5: Assume a single uniform block of vertically aligned text
        6: Assume a single uniform block of text (default)
        7: Treat the image as a single text line
        8: Treat the image as a single word
        9: Treat the image as a single word in a circle
        10: Treat the image as a single character
        11: Sparse text. Find as much text as possible in no particular order
        12: Sparse text with OSD
        13: Raw line. Treat the image as a single text line, bypassing hacks
        """
        if 0 <= psm <= 13:
            self.psm = psm
            self.config = f'--oem {self.oem} --psm {self.psm} -l {self.language}'
        else:
            self.logger.warning(f"Invalid PSM mode: {psm}")
    
    def extract_text_only(self, image: np.ndarray) -> str:
        """Extract only text content without detailed analysis."""
        if not self.is_available():
            return ""
        
        try:
            return self.pytesseract.image_to_string(image, config=self.config)
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return ""
