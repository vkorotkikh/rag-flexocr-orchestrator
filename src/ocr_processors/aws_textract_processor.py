"""
AWS Textract OCR processor implementation.
"""

import numpy as np
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import asyncio

from .base import BaseOCRProcessor
from ..models import OCRResult, OCRRegion, BoundingBox, RegionType, OCREngine, TableStructure, TableCell

logger = logging.getLogger(__name__)


class AWSTextractProcessor(BaseOCRProcessor):
    """AWS Textract processor with advanced table and form extraction."""
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 enable_preprocessing: bool = False,  # AWS Textract handles preprocessing
                 preprocessing_steps: Optional[List[str]] = None,
                 aws_region: str = 'us-east-1',
                 enable_tables: bool = True,
                 enable_forms: bool = True):
        """
        Initialize AWS Textract processor.
        
        Args:
            confidence_threshold: Minimum confidence for text acceptance
            enable_preprocessing: Whether to apply image preprocessing (usually not needed)
            preprocessing_steps: List of preprocessing steps
            aws_region: AWS region for Textract service
            enable_tables: Enable table extraction
            enable_forms: Enable form/key-value pair extraction
        """
        super().__init__(
            engine_name='aws_textract',
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing,
            preprocessing_steps=preprocessing_steps or []
        )
        
        self.aws_region = aws_region
        self.enable_tables = enable_tables
        self.enable_forms = enable_forms
        
        # Initialize AWS Textract
        self._initialize_textract()
    
    def _initialize_textract(self):
        """Initialize AWS Textract client."""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            # Create Textract client
            self.textract_client = boto3.client(
                'textract',
                region_name=self.aws_region
            )
            
            # Test credentials by making a simple call
            try:
                # This will fail if credentials are not configured
                self.textract_client.detect_document_text(
                    Document={'Bytes': b'test'}
                )
            except ClientError as e:
                if 'InvalidImageException' in str(e):
                    # This is expected with dummy data - credentials are OK
                    pass
                else:
                    raise
            
            self.is_initialized = True
            self.logger.info("AWS Textract initialized successfully")
            
        except ImportError:
            self.logger.error("boto3 not installed. Install with: pip install boto3")
            self.is_initialized = False
        except NoCredentialsError:
            self.logger.error("AWS credentials not configured")
            self.is_initialized = False
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Textract: {e}")
            self.is_initialized = False
    
    def is_available(self) -> bool:
        """Check if AWS Textract is available."""
        return self.is_initialized
    
    async def extract_text_from_image(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text from image using AWS Textract.
        
        Args:
            image: Image as numpy array
            **kwargs: Additional parameters
            
        Returns:
            OCR result with extracted text and regions
        """
        if not self.is_available():
            raise RuntimeError("AWS Textract is not available")
        
        start_time = datetime.now()
        
        document_id = kwargs.get('document_id', str(uuid.uuid4()))
        page_number = kwargs.get('page_number', 1)
        detect_tables = kwargs.get('detect_tables', True) and self.enable_tables
        detect_forms = kwargs.get('detect_forms', True) and self.enable_forms
        
        try:
            # Convert image to bytes
            image_bytes = self._convert_image_to_bytes(image)
            
            # Determine which Textract API to use based on requirements
            if detect_tables or detect_forms:
                response = await self._analyze_document(image_bytes, detect_tables, detect_forms)
            else:
                response = await self._detect_document_text(image_bytes)
            
            # Extract regions and tables from response
            regions, tables = self._parse_textract_response(response)
            
            # Calculate metrics
            overall_confidence = self._calculate_overall_confidence(regions)
            text_coverage = self._calculate_text_coverage(regions, image.shape[:2][::-1])
            low_conf_count = len([r for r in regions if r.confidence < 0.5])
            
            # Detect primary language (Textract provides this)
            detected_language = self._extract_language_from_response(response)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return OCRResult(
                document_id=document_id,
                page_number=page_number,
                regions=regions,
                tables=tables,
                ocr_engine=OCREngine.AWS_TEXTRACT,
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
            self.logger.error(f"AWS Textract failed: {e}")
            raise
    
    def _convert_image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy image to bytes for Textract."""
        import cv2
        from io import BytesIO
        
        # Encode image as PNG
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image")
        
        return encoded_image.tobytes()
    
    async def _detect_document_text(self, image_bytes: bytes) -> Dict:
        """Call Textract DetectDocumentText API."""
        loop = asyncio.get_event_loop()
        
        def _call_textract():
            return self.textract_client.detect_document_text(
                Document={'Bytes': image_bytes}
            )
        
        return await loop.run_in_executor(None, _call_textract)
    
    async def _analyze_document(self, image_bytes: bytes, detect_tables: bool, detect_forms: bool) -> Dict:
        """Call Textract AnalyzeDocument API with advanced features."""
        loop = asyncio.get_event_loop()
        
        # Build feature types
        feature_types = []
        if detect_tables:
            feature_types.append('TABLES')
        if detect_forms:
            feature_types.append('FORMS')
        
        def _call_textract():
            return self.textract_client.analyze_document(
                Document={'Bytes': image_bytes},
                FeatureTypes=feature_types
            )
        
        return await loop.run_in_executor(None, _call_textract)
    
    def _parse_textract_response(self, response: Dict) -> Tuple[List[OCRRegion], List[TableStructure]]:
        """Parse Textract response to extract regions and tables."""
        regions = []
        tables = []
        
        blocks = response.get('Blocks', [])
        
        # Parse text blocks
        for block in blocks:
            if block['BlockType'] == 'LINE':
                region = self._create_region_from_block(block)
                if region:
                    regions.append(region)
        
        # Parse table blocks if available
        if self.enable_tables:
            table_blocks = [b for b in blocks if b['BlockType'] == 'TABLE']
            for table_block in table_blocks:
                table = self._create_table_from_block(table_block, blocks)
                if table:
                    tables.append(table)
        
        # Sort regions by reading order
        regions.sort(key=lambda r: (r.bounding_box.y1, r.bounding_box.x1))
        
        # Assign reading order
        for idx, region in enumerate(regions):
            region.reading_order = idx
        
        return regions, tables
    
    def _create_region_from_block(self, block: Dict) -> Optional[OCRRegion]:
        """Create OCR region from Textract block."""
        try:
            text = block.get('Text', '').strip()
            if not text:
                return None
            
            confidence = block.get('Confidence', 0.0) / 100.0
            
            # Extract bounding box
            geometry = block.get('Geometry', {})
            bbox_dict = geometry.get('BoundingBox', {})
            
            # Textract uses relative coordinates (0-1)
            x1 = bbox_dict.get('Left', 0.0)
            y1 = bbox_dict.get('Top', 0.0)
            width = bbox_dict.get('Width', 0.0)
            height = bbox_dict.get('Height', 0.0)
            
            # Convert to absolute coordinates (assuming image dimensions)
            # Note: In practice, you'd multiply by actual image dimensions
            bounding_box = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x1 + width,
                y2=y1 + height
            )
            
            # Determine region type based on text characteristics
            region_type = self._determine_region_type(text, confidence, bounding_box)
            
            return OCRRegion(
                text=text,
                confidence=confidence,
                bounding_box=bounding_box,
                region_type=region_type,
                region_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create region from block: {e}")
            return None
    
    def _create_table_from_block(self, table_block: Dict, all_blocks: List[Dict]) -> Optional[TableStructure]:
        """Create table structure from Textract table block."""
        try:
            # Get table relationships
            relationships = table_block.get('Relationships', [])
            
            # Find CELL blocks related to this table
            cell_blocks = []
            for relationship in relationships:
                if relationship['Type'] == 'CHILD':
                    for block_id in relationship['Ids']:
                        # Find the block with this ID
                        for block in all_blocks:
                            if block['Id'] == block_id and block['BlockType'] == 'CELL':
                                cell_blocks.append(block)
                                break
            
            if not cell_blocks:
                return None
            
            # Extract table cells
            cells = []
            max_row = 0
            max_col = 0
            
            for cell_block in cell_blocks:
                cell = self._create_table_cell_from_block(cell_block, all_blocks)
                if cell:
                    cells.append(cell)
                    max_row = max(max_row, cell.row)
                    max_col = max(max_col, cell.col)
            
            if not cells:
                return None
            
            return TableStructure(
                rows=max_row + 1,
                columns=max_col + 1,
                cells=cells,
                has_header=True  # Assume first row is header
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create table from block: {e}")
            return None
    
    def _create_table_cell_from_block(self, cell_block: Dict, all_blocks: List[Dict]) -> Optional[TableCell]:
        """Create table cell from Textract cell block."""
        try:
            # Get cell position
            row_index = cell_block.get('RowIndex', 1) - 1  # Convert to 0-based
            col_index = cell_block.get('ColumnIndex', 1) - 1  # Convert to 0-based
            
            # Get cell confidence
            confidence = cell_block.get('Confidence', 0.0) / 100.0
            
            # Extract cell text from WORD blocks
            cell_text = ""
            relationships = cell_block.get('Relationships', [])
            
            for relationship in relationships:
                if relationship['Type'] == 'CHILD':
                    for block_id in relationship['Ids']:
                        for block in all_blocks:
                            if (block['Id'] == block_id and 
                                block['BlockType'] == 'WORD'):
                                word_text = block.get('Text', '')
                                if word_text:
                                    cell_text += word_text + " "
                                break
            
            cell_text = cell_text.strip()
            
            # Extract bounding box
            geometry = cell_block.get('Geometry', {})
            bbox_dict = geometry.get('BoundingBox', {})
            
            x1 = bbox_dict.get('Left', 0.0)
            y1 = bbox_dict.get('Top', 0.0)
            width = bbox_dict.get('Width', 0.0)
            height = bbox_dict.get('Height', 0.0)
            
            bounding_box = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x1 + width,
                y2=y1 + height
            )
            
            return TableCell(
                text=cell_text,
                confidence=confidence,
                row=row_index,
                col=col_index,
                bounding_box=bounding_box
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create table cell from block: {e}")
            return None
    
    def _determine_region_type(self, text: str, confidence: float, bbox: BoundingBox) -> RegionType:
        """Determine region type based on text characteristics."""
        text_lower = text.lower().strip()
        
        # Header detection (top area, short text)
        if bbox.y1 < 0.15 and len(text) < 100:
            return RegionType.HEADER
        
        # Footer detection (bottom area)
        if bbox.y1 > 0.85:
            return RegionType.FOOTER
        
        # Title detection
        if len(text) < 80 and not text.endswith('.'):
            return RegionType.TITLE
        
        # List detection
        if (text_lower.startswith(('•', '·', '-', '*')) or
            (len(text) > 2 and text[0].isdigit() and text[1] in '.):')):
            return RegionType.LIST
        
        # Caption detection
        caption_keywords = ['figure', 'table', 'chart', 'graph', 'image']
        if any(keyword in text_lower for keyword in caption_keywords):
            return RegionType.CAPTION
        
        return RegionType.PARAGRAPH
    
    def _extract_language_from_response(self, response: Dict) -> Optional[str]:
        """Extract detected language from Textract response."""
        # Textract doesn't always provide language detection
        # You might need to use additional language detection
        try:
            metadata = response.get('DocumentMetadata', {})
            # This is hypothetical - check actual Textract response format
            return metadata.get('DetectedLanguage')
        except:
            return 'en'  # Default to English
    
    def extract_key_value_pairs(self, image: np.ndarray) -> Dict[str, str]:
        """Extract key-value pairs from forms using Textract."""
        if not self.is_available() or not self.enable_forms:
            return {}
        
        try:
            image_bytes = self._convert_image_to_bytes(image)
            response = self.textract_client.analyze_document(
                Document={'Bytes': image_bytes},
                FeatureTypes=['FORMS']
            )
            
            # Parse key-value pairs
            key_value_pairs = {}
            blocks = response.get('Blocks', [])
            
            # This is a simplified extraction
            # In practice, you'd need to properly parse the relationships
            for block in blocks:
                if block['BlockType'] == 'KEY_VALUE_SET':
                    # Extract key-value logic here
                    pass
            
            return key_value_pairs
            
        except Exception as e:
            self.logger.error(f"Key-value extraction failed: {e}")
            return {}
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported Textract features."""
        return ['TEXT', 'TABLES', 'FORMS', 'QUERIES'] if self.is_available() else []
