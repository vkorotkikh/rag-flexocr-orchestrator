"""
EasyOCR processor implementation.
"""

import cv2
import numpy as np
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging

from .base import BaseOCRProcessor
from ..models import OCRResult, OCRRegion, BoundingBox, RegionType, OCREngine, TableStructure, TableCell

logger = logging.getLogger(__name__)


class EasyOCRProcessor(BaseOCRProcessor):
    """EasyOCR processor with advanced multi-language support and user-friendly interface."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 enable_preprocessing: bool = True,
                 preprocessing_steps: Optional[List[str]] = None,
                 languages: List[str] = ['en'],
                 use_gpu: bool = False,
                 detector_backend: str = 'craft',
                 recognizer_backend: str = 'crnn'):
        """
        Initialize EasyOCR processor.
        
        Args:
            confidence_threshold: Minimum confidence for text acceptance
            enable_preprocessing: Whether to apply image preprocessing
            preprocessing_steps: List of preprocessing steps
            languages: List of language codes (['en', 'zh', 'fr', etc.])
            use_gpu: Use GPU acceleration if available
            detector_backend: Text detection backend ('craft', 'dbnet')
            recognizer_backend: Text recognition backend ('crnn')
        """
        super().__init__(
            engine_name='easyocr',
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing,
            preprocessing_steps=preprocessing_steps
        )
        
        self.languages = languages if isinstance(languages, list) else [languages]
        self.use_gpu = use_gpu
        self.detector_backend = detector_backend
        self.recognizer_backend = recognizer_backend
        
        # Initialize EasyOCR
        self._initialize_easyocr()
    
    def _initialize_easyocr(self):
        """Initialize EasyOCR engine."""
        try:
            import easyocr
            
            # Initialize OCR reader
            self.ocr_reader = easyocr.Reader(
                lang_list=self.languages,
                gpu=self.use_gpu,
                model_storage_directory=None,  # Use default
                user_network_directory=None,  # Use default
                recog_network=self.recognizer_backend,
                detect_network=self.detector_backend,
                verbose=False
            )
            
            self.is_initialized = True
            self.logger.info(f"EasyOCR initialized with languages: {self.languages}")
            
        except ImportError:
            self.logger.error("EasyOCR not installed. Install with: pip install easyocr")
            self.is_initialized = False
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            self.is_initialized = False
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available."""
        return self.is_initialized
    
    async def extract_text_from_image(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text from image using EasyOCR.
        
        Args:
            image: Preprocessed image as numpy array
            **kwargs: Additional parameters
            
        Returns:
            OCR result with extracted text and regions
        """
        if not self.is_available():
            raise RuntimeError("EasyOCR is not available")
        
        start_time = datetime.now()
        
        document_id = kwargs.get('document_id', str(uuid.uuid4()))
        page_number = kwargs.get('page_number', 1)
        detect_tables = kwargs.get('detect_tables', True)
        detect_figures = kwargs.get('detect_figures', True)
        
        try:
            # Perform OCR
            ocr_results = self.ocr_reader.readtext(
                image,
                detail=1,  # Return bounding box, text, and confidence
                paragraph=False,  # Don't group into paragraphs
                width_ths=0.7,  # Text width threshold
                height_ths=0.7,  # Text height threshold
                decoder='greedy'  # Decoding method
            )
            
            # Extract regions from OCR results
            regions = self._extract_regions_from_results(ocr_results)
            
            # Simple table detection (EasyOCR doesn't have built-in structure analysis)
            tables = []
            if detect_tables:
                tables = self._detect_simple_tables(image, regions)
            
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
                ocr_engine=OCREngine.PADDLEOCR,  # We'll update the enum later
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
            self.logger.error(f"EasyOCR failed: {e}")
            raise
    
    def _extract_regions_from_results(self, ocr_results: List) -> List[OCRRegion]:
        """Extract text regions from EasyOCR results."""
        regions = []
        
        if not ocr_results:
            return regions
        
        for idx, result in enumerate(ocr_results):
            try:
                # EasyOCR result format: [bbox_points, text, confidence]
                bbox_points = result[0]
                text = result[1].strip()
                confidence = float(result[2])
                
                if not text:
                    continue
                
                # Convert polygon to bounding box
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                bounding_box = BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2)
                )
                
                # Determine region type
                region_type = self._determine_region_type(text, confidence, bounding_box)
                
                # Create region
                region = OCRRegion(
                    text=text,
                    confidence=confidence,
                    bounding_box=bounding_box,
                    region_type=region_type,
                    region_id=str(uuid.uuid4()),
                    reading_order=idx
                )
                
                regions.append(region)
                
            except (IndexError, ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse OCR result {idx}: {e}")
                continue
        
        # Sort regions by reading order (top to bottom, left to right)
        regions.sort(key=lambda r: (r.bounding_box.y1, r.bounding_box.x1))
        
        # Update reading order after sorting
        for idx, region in enumerate(regions):
            region.reading_order = idx
        
        return regions
    
    def _determine_region_type(self, text: str, confidence: float, bbox: BoundingBox) -> RegionType:
        """Determine the type of text region based on characteristics."""
        text_lower = text.lower().strip()
        
        # Header/title detection based on position and content
        if bbox.y1 < 100 and len(text) < 100:  # Top area, short text
            return RegionType.HEADER
        
        # Footer detection
        if bbox.y1 > 800:  # Bottom area (adjust based on image size)
            return RegionType.FOOTER
        
        # Title detection (short text, potentially larger font)
        if len(text) < 60 and not text.endswith('.'):
            return RegionType.TITLE
        
        # List detection
        list_indicators = ['•', '·', '-', '*', '○', '■', '□']
        if (any(text_lower.startswith(indicator) for indicator in list_indicators) or
            (len(text) > 2 and text[0].isdigit() and text[1] in '.):')):
            return RegionType.LIST
        
        # Caption detection
        caption_keywords = ['figure', 'table', 'chart', 'graph', 'image', 'fig.', 'tab.', 'caption']
        if any(keyword in text_lower for keyword in caption_keywords):
            return RegionType.CAPTION
        
        # Default to paragraph
        return RegionType.PARAGRAPH
    
    def _detect_simple_tables(self, image: np.ndarray, regions: List[OCRRegion]) -> List[TableStructure]:
        """Simple table detection for EasyOCR (basic implementation)."""
        tables = []
        
        try:
            # Basic table detection using region alignment
            # Group regions that are horizontally and vertically aligned
            aligned_groups = self._find_aligned_regions(regions)
            
            for group in aligned_groups:
                if self._looks_like_table(group):
                    table = self._create_table_from_regions(group)
                    if table:
                        tables.append(table)
            
        except Exception as e:
            self.logger.warning(f"Simple table detection failed: {e}")
        
        return tables
    
    def _find_aligned_regions(self, regions: List[OCRRegion]) -> List[List[OCRRegion]]:
        """Find groups of aligned regions that might form tables."""
        if len(regions) < 4:  # Need at least 4 regions for a table
            return []
        
        # Sort regions by y-position
        sorted_regions = sorted(regions, key=lambda r: r.bounding_box.y1)
        
        # Group regions into rows based on vertical alignment
        rows = []
        current_row = [sorted_regions[0]]
        row_tolerance = 20  # pixels
        
        for region in sorted_regions[1:]:
            if abs(region.bounding_box.y1 - current_row[-1].bounding_box.y1) <= row_tolerance:
                current_row.append(region)
            else:
                if len(current_row) >= 2:  # Row must have at least 2 columns
                    rows.append(current_row)
                current_row = [region]
        
        if len(current_row) >= 2:
            rows.append(current_row)
        
        # Return groups that look like tables
        return [row for row in rows if len(row) >= 2] if len(rows) >= 2 else []
    
    def _looks_like_table(self, region_group: List[OCRRegion]) -> bool:
        """Check if a group of regions looks like a table."""
        if len(region_group) < 4:
            return False
        
        # Check for consistent spacing and alignment
        x_positions = [r.bounding_box.x1 for r in region_group]
        
        # Simple heuristic: if regions have similar x-positions, might be a table
        unique_x = len(set(round(x, -1) for x in x_positions))  # Round to nearest 10
        return unique_x >= 2  # At least 2 columns
    
    def _create_table_from_regions(self, regions: List[OCRRegion]) -> Optional[TableStructure]:
        """Create a simple table structure from aligned regions."""
        if not regions:
            return None
        
        # Create simple table cells
        cells = []
        for i, region in enumerate(regions):
            cell = TableCell(
                text=region.text,
                confidence=region.confidence,
                row=i // 2,  # Simple row assignment
                col=i % 2,   # Simple column assignment
                bounding_box=region.bounding_box
            )
            cells.append(cell)
        
        # Estimate table dimensions
        max_row = max(cell.row for cell in cells) + 1 if cells else 1
        max_col = max(cell.col for cell in cells) + 1 if cells else 1
        
        return TableStructure(
            rows=max_row,
            columns=max_col,
            cells=cells,
            has_header=True  # Assume first row is header
        )
    
    def _extract_table_structure(self, table_result: Dict, regions: List[OCRRegion]) -> Optional[TableStructure]:
        """Extract table structure from PaddleOCR structure analysis result."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to parse the HTML table structure returned by PaddleOCR
            
            # Get table bounding box
            bbox = table_result.get('bbox', [])
            if len(bbox) != 4:
                return None
            
            table_bbox = BoundingBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3])
            )
            
            # Find regions within table bounds
            table_regions = []
            for region in regions:
                if self._is_region_in_table(region.bounding_box, table_bbox):
                    table_regions.append(region)
            
            if not table_regions:
                return None
            
            # Simplified table cell extraction
            # In practice, you'd parse the actual table structure
            cells = []
            for idx, region in enumerate(table_regions):
                cell = TableCell(
                    text=region.text,
                    confidence=region.confidence,
                    row=idx // 3,  # Simplified: assume 3 columns
                    col=idx % 3,
                    bounding_box=region.bounding_box
                )
                cells.append(cell)
            
            # Estimate table dimensions
            max_row = max(cell.row for cell in cells) + 1 if cells else 1
            max_col = max(cell.col for cell in cells) + 1 if cells else 1
            
            return TableStructure(
                rows=max_row,
                columns=max_col,
                cells=cells,
                has_header=True  # Simplified assumption
            )
            
        except Exception as e:
            self.logger.warning(f"Table structure extraction failed: {e}")
            return None
    
    def _is_region_in_table(self, region_bbox: BoundingBox, table_bbox: BoundingBox) -> bool:
        """Check if a region is within a table's bounds."""
        return (region_bbox.x1 >= table_bbox.x1 and
                region_bbox.y1 >= table_bbox.y1 and
                region_bbox.x2 <= table_bbox.x2 and
                region_bbox.y2 <= table_bbox.y2)
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of extracted text."""
        try:
            from langdetect import detect
            if text and len(text.strip()) > 20:
                detected = detect(text)
                # Map to EasyOCR language codes
                lang_map = {
                    'en': 'en',
                    'zh': 'ch_sim',
                    'zh-cn': 'ch_sim', 
                    'fr': 'fr',
                    'de': 'de',
                    'ja': 'ja',
                    'ko': 'ko',
                    'es': 'es',
                    'pt': 'pt',
                    'ru': 'ru'
                }
                return lang_map.get(detected, detected)
        except:
            pass
        return self.languages[0] if self.languages else 'en'
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # EasyOCR supported languages (most common ones)
        return [
            'en', 'ch_sim', 'ch_tra', 'fr', 'de', 'ja', 'ko', 
            'es', 'pt', 'ru', 'ar', 'hi', 'th', 'vi', 'it',
            'nl', 'pl', 'sv', 'da', 'no', 'fi', 'tr', 'cs',
            'sk', 'hr', 'hu', 'bg', 'lt', 'lv', 'et', 'mt',
            'cy', 'eu', 'ca', 'gl', 'is', 'mk', 'ro', 'sq',
            'bs', 'sr', 'sl', 'uk', 'be', 'mn', 'ne', 'bn',
            'as', 'or', 'te', 'kn', 'ml', 'ta', 'mr', 'gu',
            'pa', 'ur', 'fa', 'ug', 'bo', 'dz', 'my', 'lo',
            'km', 'ka', 'am', 'ti', 'hy', 'az', 'kk', 'ky',
            'uz', 'tj', 'tg', 'mn_cyrl', 'si', 'my_zaw'
        ]
    
    def set_languages(self, languages: List[str]):
        """Set OCR languages."""
        supported = self.get_supported_languages()
        valid_languages = [lang for lang in languages if lang in supported]
        
        if valid_languages:
            self.languages = valid_languages
            # Reinitialize with new languages
            self._initialize_easyocr()
        else:
            self.logger.warning(f"No supported languages in {languages}")
    
    def set_language(self, language: str):
        """Set single OCR language (for compatibility)."""
        self.set_languages([language])
    
    def extract_text_only(self, image: np.ndarray) -> str:
        """Extract only text content without detailed analysis."""
        if not self.is_available():
            return ""
        
        try:
            results = self.ocr_reader.readtext(image, detail=0)  # detail=0 returns only text
            if results:
                return ' '.join(results)
            return ""
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            return ""
    
    def visualize_results(self, image: np.ndarray, ocr_results: List) -> np.ndarray:
        """Visualize OCR results on the image."""
        if not self.is_available() or not ocr_results:
            return image
        
        try:
            import cv2
            
            # Create a copy of the image for visualization
            vis_image = image.copy()
            
            for result in ocr_results:
                bbox_points = result[0]
                text = result[1]
                confidence = result[2]
                
                # Convert bbox points to integer coordinates
                points = np.array(bbox_points, dtype=np.int32)
                
                # Draw bounding box
                cv2.polylines(vis_image, [points], True, (0, 255, 0), 2)
                
                # Draw text and confidence
                x, y = int(points[0][0]), int(points[0][1])
                label = f"{text} ({confidence:.2f})"
                cv2.putText(vis_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            return vis_image
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
            return image
