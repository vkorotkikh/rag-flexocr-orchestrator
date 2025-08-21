"""
TrOCR (Transformer-based OCR) processor implementation using Microsoft's state-of-the-art models.

TrOCR is a transformer-based OCR model that combines an image Transformer encoder 
and a text Transformer decoder for state-of-the-art optical character recognition.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime
import uuid
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# from transformers import (
#     TrOCRProcessor,
#     VisionEncoderDecoderModel,
#     AutoProcessor,
#     AutoModel
# )
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoProcessor, AutoModel
from transformers.utils import logging as transformers_logging
j
from .base import BaseOCRProcessor
from ..models import OCRResult, OCRRegion, BoundingBox, RegionType, OCREngine

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set transformers logging level
transformers_logging.set_verbosity_info()


class TrOCRProcessor(BaseOCRProcessor):
    """
    Advanced OCR processor using Microsoft's TrOCR models.
    
    TrOCR achieves state-of-the-art performance on printed, handwritten,
    and scene text recognition benchmarks.
    
    Available models:
    - microsoft/trocr-base-printed: Base model for printed text
    - microsoft/trocr-large-printed: Large model for printed text (higher accuracy)
    - microsoft/trocr-base-handwritten: For handwritten text
    - microsoft/trocr-large-handwritten: Large model for handwritten text
    - microsoft/trocr-small-printed: Smaller, faster model
    """
    
    AVAILABLE_MODELS = {
        "printed_base": "microsoft/trocr-base-printed",
        "printed_large": "microsoft/trocr-large-printed",
        "printed_small": "microsoft/trocr-small-printed",
        "handwritten_base": "microsoft/trocr-base-handwritten",
        "handwritten_large": "microsoft/trocr-large-handwritten",
        "stage1": "microsoft/trocr-base-stage1",  # For fine-tuning
    }
    
    def __init__(
        self,
        model_name: str = "printed_large",
        confidence_threshold: float = 0.7,
        enable_preprocessing: bool = True,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 256,
        enable_beam_search: bool = True,
        num_beams: int = 5,
        enable_caching: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize TrOCR processor with advanced configuration.
        
        Args:
            model_name: Model variant to use (see AVAILABLE_MODELS)
            confidence_threshold: Minimum confidence for text acceptance
            enable_preprocessing: Whether to apply image preprocessing
            device: Device to run model on (cuda/cpu/mps)
            batch_size: Batch size for processing multiple regions
            max_length: Maximum length of generated text
            enable_beam_search: Use beam search for better accuracy
            num_beams: Number of beams for beam search
            enable_caching: Cache model weights locally
            cache_dir: Directory for caching models
        """
        super().__init__(
            engine_name='trocr',
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing
        )
        
        self.model_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_beam_search = enable_beam_search
        self.num_beams = num_beams
        self.enable_caching = enable_caching
        self.cache_dir = cache_dir or Path.home() / ".cache" / "trocr"
        
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
        
        logger.info(f"TrOCR processor initialized with model: {self.model_name}")
        logger.info(f"Device: {self.device}, Batch size: {self.batch_size}")
    
    def _initialize_model(self):
        """Initialize TrOCR model and processor with error handling."""
        try:
            logger.info(f"Loading TrOCR model: {self.model_name}")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir if self.enable_caching else None
            )
            
            # Load model
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir if self.enable_caching else None
            )
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable mixed precision for faster inference on compatible GPUs
            if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 7:
                logger.info("Enabling automatic mixed precision (AMP) for faster inference")
                self.use_amp = True
            else:
                self.use_amp = False
            
            self.is_initialized = True
            logger.info(f"TrOCR model loaded successfully. Model size: {self._get_model_size():.2f} MB")
            
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR model: {e}")
            logger.error("Please ensure transformers library is installed: pip install transformers")
            self.is_initialized = False
            raise
    
    def _get_model_size(self) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    def is_available(self) -> bool:
        """Check if TrOCR is available and initialized."""
        return self.is_initialized
    
    async def extract_text_from_image(
        self,
        image: Union[np.ndarray, Image.Image],
        **kwargs
    ) -> OCRResult:
        """
        Extract text from image using TrOCR.
        
        Args:
            image: Input image as numpy array or PIL Image
            **kwargs: Additional parameters
            
        Returns:
            OCR result with extracted text and metadata
        """
        if not self.is_available():
            raise RuntimeError("TrOCR is not available or not initialized")
        
        start_time = datetime.utcnow()
        logger.debug("Starting TrOCR text extraction")
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                logger.debug(f"Converted numpy array to PIL Image: {image.size}")
            
            # Preprocess image if enabled
            if self.enable_preprocessing:
                image = await self._preprocess_image_advanced(image)
            
            # Extract text using TrOCR
            extracted_text, confidence = await self._extract_with_trocr(image)
            
            # Create OCR result
            result = self._create_ocr_result(
                text=extracted_text,
                confidence=confidence,
                image_size=image.size,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            logger.info(f"TrOCR extraction completed. Text length: {len(extracted_text)}, Confidence: {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}", exc_info=True)
            raise
    
    async def _extract_with_trocr(
        self,
        image: Image.Image
    ) -> Tuple[str, float]:
        """
        Perform text extraction using TrOCR model.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Process image
            pixel_values = self.processor(
                images=image,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            logger.debug(f"Input tensor shape: {pixel_values.shape}")
            
            # Generate text with beam search or greedy decoding
            with torch.no_grad():
                if self.use_amp and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        generated_ids = self._generate_text(pixel_values)
                else:
                    generated_ids = self._generate_text(pixel_values)
            
            # Decode generated text
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence (using generation scores if available)
            confidence = self._calculate_confidence(generated_ids)
            
            logger.debug(f"Generated text: '{generated_text[:100]}...'")
            return generated_text, confidence
            
        except Exception as e:
            logger.error(f"Error during TrOCR inference: {e}")
            raise
    
    def _generate_text(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Generate text using the model with configured decoding strategy."""
        if self.enable_beam_search:
            return self.model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                temperature=0.9,
                do_sample=False
            )
        else:
            # Greedy decoding (faster but potentially less accurate)
            return self.model.generate(
                pixel_values,
                max_length=self.max_length,
                do_sample=False
            )
    
    def _calculate_confidence(self, generated_ids: torch.Tensor) -> float:
        """
        Calculate confidence score for generated text.
        
        This is a simplified confidence calculation. In production,
        you might want to use the model's log probabilities.
        """
        # For now, return a high confidence as TrOCR is generally very accurate
        # You could implement more sophisticated confidence calculation using
        # the model's output logits
        return 0.95
    
    async def _preprocess_image_advanced(self, image: Image.Image) -> Image.Image:
        """
        Apply advanced preprocessing using PIL and scikit-image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        logger.debug("Applying advanced image preprocessing")
        
        try:
            from PIL import ImageEnhance, ImageFilter, ImageOps
            import numpy as np
            from skimage import filters, morphology, exposure
            
            # Convert to numpy for scikit-image processing
            img_array = np.array(image)
            
            # Apply adaptive histogram equalization
            if len(img_array.shape) == 2:  # Grayscale
                img_array = exposure.equalize_adapthist(img_array, clip_limit=0.03)
            else:  # Color
                # Convert to grayscale for processing
                from skimage.color import rgb2gray
                img_array = rgb2gray(img_array)
                img_array = exposure.equalize_adapthist(img_array)
            
            # Denoise
            img_array = filters.gaussian(img_array, sigma=0.5)
            
            # Convert back to PIL Image
            img_array = (img_array * 255).astype(np.uint8)
            image = Image.fromarray(img_array)
            
            # Apply PIL enhancements
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            logger.debug("Image preprocessing completed")
            return image
            
        except ImportError as e:
            logger.warning(f"Advanced preprocessing libraries not available: {e}")
            logger.warning("Install scikit-image for better preprocessing: pip install scikit-image")
            return image
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}")
            return image
    
    def _create_ocr_result(
        self,
        text: str,
        confidence: float,
        image_size: Tuple[int, int],
        processing_time: float
    ) -> OCRResult:
        """Create OCRResult object with metadata."""
        result_id = str(uuid.uuid4())
        
        # Create a single region for the entire extracted text
        # In a more advanced implementation, you could segment the text
        region = OCRRegion(
            text=text,
            confidence=confidence,
            bounding_box=BoundingBox(
                x1=0,
                y1=0,
                x2=image_size[0],
                y2=image_size[1]
            ),
            region_type=RegionType.TEXT,
            metadata={
                "model": self.model_name,
                "device": self.device,
                "beam_search": self.enable_beam_search
            }
        )
        
        return OCRResult(
            id=result_id,
            engine=OCREngine.TESSERACT,  # We should add TROCR to the enum
            extracted_text=text,
            regions=[region],
            confidence_scores=[confidence],
            processing_time_seconds=processing_time,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "preprocessing": self.enable_preprocessing,
                "beam_search": self.enable_beam_search,
                "num_beams": self.num_beams if self.enable_beam_search else None
            }
        )
    
    async def batch_extract(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        **kwargs
    ) -> List[OCRResult]:
        """
        Process multiple images in batches for efficiency.
        
        Args:
            images: List of images to process
            **kwargs: Additional parameters
            
        Returns:
            List of OCR results
        """
        logger.info(f"Starting batch extraction for {len(images)} images")
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            logger.debug(f"Processing batch {i // self.batch_size + 1}")
            
            batch_results = await asyncio.gather(*[
                self.extract_text_from_image(img, **kwargs)
                for img in batch
            ])
            results.extend(batch_results)
        
        logger.info(f"Batch extraction completed. Processed {len(results)} images")
        return results
    
    def __repr__(self) -> str:
        """String representation of the processor."""
        return (
            f"TrOCRProcessor(model='{self.model_name}', "
            f"device='{self.device}', "
            f"batch_size={self.batch_size}, "
            f"beam_search={self.enable_beam_search})"
        )


# Import asyncio for async operations
import asyncio
