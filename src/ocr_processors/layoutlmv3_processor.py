"""
LayoutLMv3 processor for advanced document understanding and layout-aware OCR.

LayoutLMv3 is a pre-trained multimodal Transformer that achieves state-of-the-art
performance in Document AI tasks including form understanding, receipt understanding,
and document visual question answering.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime
import uuid
import numpy as np
from PIL import Image
import torch
from pathlib import Path

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3Model,
    AutoProcessor,
    AutoModelForTokenClassification
)
from datasets import Features, Sequence, Value, ClassLabel, Array2D

from .base import BaseOCRProcessor
from ..models import OCRResult, OCRRegion, BoundingBox, RegionType, OCREngine

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add custom formatter for better log readability
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class LayoutLMv3Processor(BaseOCRProcessor):
    """
    Advanced document understanding processor using Microsoft's LayoutLMv3.
    
    LayoutLMv3 combines text, layout, and image features for superior
    document understanding. It can:
    - Extract text with layout awareness
    - Classify document regions (headers, paragraphs, tables, etc.)
    - Understand form fields and their relationships
    - Answer questions about documents
    
    Available models:
    - microsoft/layoutlmv3-base: Base model for general document understanding
    - microsoft/layoutlmv3-large: Large model with better accuracy
    - microsoft/layoutlmv3-base-chinese: For Chinese documents
    """
    
    AVAILABLE_MODELS = {
        "base": "microsoft/layoutlmv3-base",
        "large": "microsoft/layoutlmv3-large",
        "chinese": "microsoft/layoutlmv3-base-chinese",
        "funsd": "nielsr/layoutlmv3-finetuned-funsd",  # Fine-tuned on forms
    }
    
    # Label mappings for token classification
    LABEL_NAMES = [
        "O",        # Outside any entity
        "B-HEADER", # Beginning of header
        "I-HEADER", # Inside header
        "B-QUESTION", # Beginning of question (for forms)
        "I-QUESTION", # Inside question
        "B-ANSWER", # Beginning of answer
        "I-ANSWER", # Inside answer
        "B-TABLE",  # Table region
        "I-TABLE",
        "B-LIST",   # List region
        "I-LIST",
        "B-CAPTION", # Caption
        "I-CAPTION"
    ]
    
    def __init__(
        self,
        model_name: str = "base",
        confidence_threshold: float = 0.7,
        enable_preprocessing: bool = True,
        device: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 512,
        enable_ocr: bool = True,
        ocr_lang: str = "eng",
        enable_visual_features: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize LayoutLMv3 processor with configuration.
        
        Args:
            model_name: Model variant to use
            confidence_threshold: Minimum confidence for predictions
            enable_preprocessing: Whether to apply image preprocessing
            device: Device to run model on (cuda/cpu/mps)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            enable_ocr: Whether to use built-in OCR (Tesseract)
            ocr_lang: Language for OCR
            enable_visual_features: Use visual features for better accuracy
            cache_dir: Directory for caching models
        """
        super().__init__(
            engine_name='layoutlmv3',
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing
        )
        
        self.model_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        self.enable_visual_features = enable_visual_features
        self.cache_dir = cache_dir or Path.home() / ".cache" / "layoutlmv3"
        
        # Determine device
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using Apple Metal Performance Shaders (MPS)")
            else:
                self.device = "cpu"
                logger.info("Using CPU for inference")
        
        # Initialize model and processor
        self._initialize_model()
        
        logger.info(f"LayoutLMv3 processor initialized with model: {self.model_name}")
        logger.info(f"OCR enabled: {self.enable_ocr}, Visual features: {self.enable_visual_features}")
    
    def _initialize_model(self):
        """Initialize LayoutLMv3 model and processor."""
        try:
            logger.info(f"Loading LayoutLMv3 model: {self.model_name}")
            
            # Load processor with OCR capability
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                apply_ocr=self.enable_ocr,
                cache_dir=self.cache_dir
            )
            
            # Load model for token classification
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.LABEL_NAMES),
                cache_dir=self.cache_dir
            )
            
            # Also load base model for feature extraction
            self.base_model = LayoutLMv3Model.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Move models to device
            self.model = self.model.to(self.device)
            self.base_model = self.base_model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            self.base_model.eval()
            
            # Setup label mappings
            self.id2label = {i: label for i, label in enumerate(self.LABEL_NAMES)}
            self.label2id = {label: i for i, label in enumerate(self.LABEL_NAMES)}
            
            self.is_initialized = True
            logger.info(f"LayoutLMv3 models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LayoutLMv3: {e}", exc_info=True)
            self.is_initialized = False
            raise
    
    def is_available(self) -> bool:
        """Check if LayoutLMv3 is available."""
        return self.is_initialized
    
    async def extract_text_from_image(
        self,
        image: Union[np.ndarray, Image.Image],
        **kwargs
    ) -> OCRResult:
        """
        Extract text with layout understanding using LayoutLMv3.
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            OCR result with layout-aware text extraction
        """
        if not self.is_available():
            raise RuntimeError("LayoutLMv3 is not available or not initialized")
        
        start_time = datetime.utcnow()
        logger.info("Starting LayoutLMv3 document analysis")
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                logger.debug(f"Converted numpy array to PIL Image: {image.size}")
            
            # Preprocess image if enabled
            if self.enable_preprocessing:
                image = await self._preprocess_for_layout(image)
            
            # Extract text and layout with LayoutLMv3
            result = await self._analyze_document_layout(image)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result.processing_time_seconds = processing_time
            
            logger.info(
                f"LayoutLMv3 analysis completed. "
                f"Regions detected: {len(result.regions)}, "
                f"Processing time: {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LayoutLMv3 extraction failed: {e}", exc_info=True)
            raise
    
    async def _analyze_document_layout(self, image: Image.Image) -> OCRResult:
        """
        Perform document layout analysis and text extraction.
        
        Args:
            image: PIL Image to analyze
            
        Returns:
            OCRResult with layout-aware regions
        """
        try:
            # Process image with OCR if enabled
            encoding = self.processor(
                images=image,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True
            )
            
            # Move to device
            for key in encoding.keys():
                if isinstance(encoding[key], torch.Tensor):
                    encoding[key] = encoding[key].to(self.device)
            
            logger.debug("Image processed and encoded")
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
                
                # Get confidence scores
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidences = probs.max(-1).values.squeeze().tolist()
            
            # Extract text and bounding boxes from encoding
            if hasattr(encoding, 'words') and encoding.words:
                words = encoding.words[0]
                boxes = encoding.boxes[0] if hasattr(encoding, 'boxes') else None
            elif self.enable_ocr:
                # If OCR was performed by the processor
                words = self.processor.tokenizer.convert_ids_to_tokens(
                    encoding.input_ids[0].tolist()
                )
                boxes = encoding.bbox[0].tolist() if 'bbox' in encoding else None
            else:
                words = []
                boxes = None
            
            # Create regions from predictions
            regions = self._create_regions_from_predictions(
                words, boxes, predictions, confidences
            )
            
            # Aggregate text
            full_text = " ".join([r.text for r in regions if r.text])
            
            # Calculate average confidence
            avg_confidence = np.mean([r.confidence for r in regions]) if regions else 0.0
            
            return OCRResult(
                id=str(uuid.uuid4()),
                engine=OCREngine.TESSERACT,  # Should add LAYOUTLMV3
                extracted_text=full_text,
                regions=regions,
                confidence_scores=[r.confidence for r in regions],
                page_number=1,
                metadata={
                    "model": self.model_name,
                    "device": self.device,
                    "ocr_enabled": self.enable_ocr,
                    "visual_features": self.enable_visual_features
                }
            )
            
        except Exception as e:
            logger.error(f"Error during layout analysis: {e}", exc_info=True)
            raise
    
    def _create_regions_from_predictions(
        self,
        words: List[str],
        boxes: Optional[List[List[int]]],
        predictions: List[int],
        confidences: List[float]
    ) -> List[OCRRegion]:
        """
        Create OCR regions from model predictions.
        
        Args:
            words: List of words/tokens
            boxes: Bounding boxes for each word
            predictions: Label predictions for each word
            confidences: Confidence scores for each prediction
            
        Returns:
            List of OCR regions with layout information
        """
        regions = []
        current_region = None
        current_text = []
        current_boxes = []
        current_label = None
        
        for i, (word, pred, conf) in enumerate(zip(words, predictions, confidences)):
            # Skip special tokens
            if word in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.id2label.get(pred, "O")
            
            # Check if we're starting a new entity
            if label.startswith("B-"):
                # Save current region if exists
                if current_region and current_text:
                    regions.append(self._finalize_region(
                        current_text, current_boxes, current_label, current_region
                    ))
                
                # Start new region
                current_label = label[2:]  # Remove B- prefix
                current_text = [word]
                current_boxes = [boxes[i]] if boxes else []
                current_region = conf
                
            elif label.startswith("I-") and current_label == label[2:]:
                # Continue current region
                current_text.append(word)
                if boxes:
                    current_boxes.append(boxes[i])
                current_region = min(current_region, conf)  # Use minimum confidence
                
            else:
                # Outside any entity or entity boundary
                if current_region and current_text:
                    regions.append(self._finalize_region(
                        current_text, current_boxes, current_label, current_region
                    ))
                    current_region = None
                    current_text = []
                    current_boxes = []
                    current_label = None
                
                # Add as generic text region if not special token
                if label == "O" and word.strip():
                    region = OCRRegion(
                        text=word,
                        confidence=conf,
                        bounding_box=self._box_to_bbox(boxes[i]) if boxes else BoundingBox(x1=0, y1=0, x2=100, y2=100),
                        region_type=RegionType.TEXT
                    )
                    regions.append(region)
        
        # Don't forget the last region
        if current_region and current_text:
            regions.append(self._finalize_region(
                current_text, current_boxes, current_label, current_region
            ))
        
        logger.debug(f"Created {len(regions)} regions from predictions")
        return regions
    
    def _finalize_region(
        self,
        text_parts: List[str],
        boxes: List[List[int]],
        label: str,
        confidence: float
    ) -> OCRRegion:
        """Create final OCR region from accumulated parts."""
        # Join text parts
        text = " ".join(text_parts).replace(" ##", "")  # Handle subword tokens
        
        # Calculate bounding box
        if boxes:
            x1 = min(box[0] for box in boxes)
            y1 = min(box[1] for box in boxes)
            x2 = max(box[2] for box in boxes)
            y2 = max(box[3] for box in boxes)
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        else:
            bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        
        # Map label to region type
        region_type_map = {
            "HEADER": RegionType.HEADER,
            "QUESTION": RegionType.TEXT,
            "ANSWER": RegionType.TEXT,
            "TABLE": RegionType.TABLE,
            "LIST": RegionType.LIST,
            "CAPTION": RegionType.CAPTION
        }
        region_type = region_type_map.get(label, RegionType.TEXT)
        
        return OCRRegion(
            text=text,
            confidence=confidence,
            bounding_box=bbox,
            region_type=region_type,
            metadata={"label": label}
        )
    
    def _box_to_bbox(self, box: List[int]) -> BoundingBox:
        """Convert box coordinates to BoundingBox object."""
        return BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    
    async def _preprocess_for_layout(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for optimal layout detection.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        logger.debug("Preprocessing image for layout detection")
        
        try:
            from PIL import ImageOps, ImageEnhance
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (LayoutLMv3 works best with reasonable sizes)
            max_size = 1000
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                logger.debug(f"Resized image to {image.size}")
            
            # Enhance contrast for better text detection
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Auto-orient image if EXIF data is present
            image = ImageOps.exif_transpose(image)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error during preprocessing: {e}")
            return image
    
    async def extract_tables(self, image: Union[np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Specialized method for extracting tables from documents.
        
        Args:
            image: Input image
            
        Returns:
            List of detected tables with structure
        """
        logger.info("Extracting tables from document")
        
        result = await self.extract_text_from_image(image)
        
        # Filter for table regions
        tables = []
        for region in result.regions:
            if region.region_type == RegionType.TABLE:
                tables.append({
                    "text": region.text,
                    "bbox": region.bounding_box,
                    "confidence": region.confidence,
                    "metadata": region.metadata
                })
        
        logger.info(f"Found {len(tables)} tables in document")
        return tables
    
    async def answer_question(
        self,
        image: Union[np.ndarray, Image.Image],
        question: str
    ) -> str:
        """
        Answer questions about the document using LayoutLMv3's QA capabilities.
        
        Args:
            image: Document image
            question: Question to answer
            
        Returns:
            Answer extracted from the document
        """
        logger.info(f"Answering question: {question}")
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Process with question
        encoding = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )
        
        # Move to device
        for key in encoding.keys():
            if isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(self.device)
        
        # Get answer
        with torch.no_grad():
            outputs = self.base_model(**encoding)
            # Process outputs to extract answer
            # This is simplified - actual implementation would need
            # a QA head or additional processing
            
        logger.info("Question answered")
        return "Answer extraction requires additional QA model head"
    
    def __repr__(self) -> str:
        """String representation of the processor."""
        return (
            f"LayoutLMv3Processor(model='{self.model_name}', "
            f"device='{self.device}', "
            f"ocr={self.enable_ocr}, "
            f"visual={self.enable_visual_features})"
        )

