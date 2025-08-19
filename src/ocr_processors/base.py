"""
Base OCR processor interface and image preprocessing utilities.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import logging

from ..models import OCRResult, OCRDocument, BoundingBox, OCRRegion, OCREngine, RegionType

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Advanced image preprocessing for optimal OCR results."""
    
    def __init__(self):
        self.preprocessing_steps = {
            'noise_reduction': self._apply_noise_reduction,
            'contrast_enhancement': self._apply_contrast_enhancement,
            'deskew': self._apply_deskewing,
            'binarization': self._apply_binarization,
            'rotation_correction': self._apply_rotation_correction,
            'scaling': self._apply_scaling,
            'sharpening': self._apply_sharpening
        }
    
    def preprocess_image(
        self, 
        image: Union[np.ndarray, Image.Image, bytes], 
        steps: List[str] = None,
        target_dpi: int = 300
    ) -> np.ndarray:
        """
        Apply preprocessing steps to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array, PIL Image, or bytes
            steps: List of preprocessing steps to apply
            target_dpi: Target DPI for optimal OCR
            
        Returns:
            Preprocessed image as numpy array
        """
        if steps is None:
            steps = ['noise_reduction', 'contrast_enhancement', 'deskew', 'binarization']
        
        # Convert input to numpy array
        img_array = self._convert_to_numpy(image)
        
        logger.info(f"Starting image preprocessing with steps: {steps}")
        
        for step in steps:
            if step in self.preprocessing_steps:
                try:
                    img_array = self.preprocessing_steps[step](img_array)
                    logger.debug(f"Applied preprocessing step: {step}")
                except Exception as e:
                    logger.warning(f"Failed to apply preprocessing step {step}: {e}")
                    continue
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        
        # Ensure optimal resolution
        if target_dpi and target_dpi > 0:
            img_array = self._ensure_optimal_resolution(img_array, target_dpi)
        
        return img_array
    
    def _convert_to_numpy(self, image: Union[np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """Convert various image formats to numpy array."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, bytes):
            # Convert bytes to PIL Image first
            from io import BytesIO
            pil_image = Image.open(BytesIO(image))
            return np.array(pil_image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filter."""
        if len(image.shape) == 3:
            # Color image
            return cv2.bilateralFilter(image, 9, 75, 75)
        else:
            # Grayscale image
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def _apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _apply_deskewing(self, image: np.ndarray) -> np.ndarray:
        """Correct skew angle using Hough line transform."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:  # Only consider reasonable skew angles
                    angles.append(angle)
            
            if angles:
                avg_angle = np.mean(angles)
                if abs(avg_angle) > 0.5:  # Only correct if skew is significant
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    return cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def _apply_binarization(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try different binarization methods and choose the best
        methods = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        ]
        
        # Choose method with best contrast (simple heuristic)
        best_method = methods[0]
        best_contrast = np.std(best_method)
        
        for method in methods[1:]:
            contrast = np.std(method)
            if contrast > best_contrast:
                best_contrast = contrast
                best_method = method
        
        return best_method
    
    def _apply_rotation_correction(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct rotation using text line orientation."""
        # This is a simplified version - more sophisticated methods could be implemented
        return self._apply_deskewing(image)
    
    def _apply_scaling(self, image: np.ndarray) -> np.ndarray:
        """Scale image to optimal size for OCR."""
        height, width = image.shape[:2]
        
        # Target height for optimal OCR (around 1000-2000 pixels)
        target_height = 1500
        
        if height < target_height * 0.5:
            # Upscale if too small
            scale_factor = target_height / height
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        elif height > target_height * 2:
            # Downscale if too large
            scale_factor = target_height / height
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter to enhance text clarity."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _ensure_optimal_resolution(self, image: np.ndarray, target_dpi: int) -> np.ndarray:
        """Ensure image has optimal resolution for OCR."""
        # This is a simplified version - in practice, you'd need image DPI metadata
        # For now, we'll ensure minimum dimensions
        height, width = image.shape[:2]
        
        # Assume current DPI and calculate target dimensions
        min_width = int(width * target_dpi / 150)  # Assuming 150 DPI as baseline
        min_height = int(height * target_dpi / 150)
        
        if width < min_width or height < min_height:
            scale_x = min_width / width if width < min_width else 1.0
            scale_y = min_height / height if height < min_height else 1.0
            scale = max(scale_x, scale_y)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return image


class BaseOCRProcessor(ABC):
    """Base class for all OCR processors."""
    
    def __init__(self, 
                 engine_name: str,
                 confidence_threshold: float = 0.7,
                 enable_preprocessing: bool = True,
                 preprocessing_steps: Optional[List[str]] = None):
        self.engine_name = engine_name
        self.confidence_threshold = confidence_threshold
        self.enable_preprocessing = enable_preprocessing
        self.preprocessing_steps = preprocessing_steps or [
            'noise_reduction', 'contrast_enhancement', 'deskew', 'binarization'
        ]
        self.preprocessor = ImagePreprocessor()
        self.logger = logging.getLogger(f"{__name__}.{engine_name}")
    
    @abstractmethod
    async def extract_text_from_image(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text from a preprocessed image.
        
        Args:
            image: Preprocessed image as numpy array
            **kwargs: Engine-specific parameters
            
        Returns:
            OCR result with extracted text and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the OCR engine is available and properly configured."""
        pass
    
    async def process_document(self, document: OCRDocument) -> List[OCRResult]:
        """
        Process a complete document and return OCR results for all pages.
        
        Args:
            document: OCR document to process
            
        Returns:
            List of OCR results, one per page
        """
        if not self.is_available():
            raise RuntimeError(f"OCR engine {self.engine_name} is not available")
        
        self.logger.info(f"Starting OCR processing for document: {document.id}")
        
        # Extract images from document
        images = await self._extract_images_from_document(document)
        
        results = []
        for page_num, image in enumerate(images, 1):
            try:
                # Preprocess image if enabled
                if self.enable_preprocessing and document.processing_config.enhance_quality:
                    processed_image = self.preprocessor.preprocess_image(
                        image, 
                        steps=self.preprocessing_steps,
                        target_dpi=document.processing_config.max_resolution
                    )
                else:
                    processed_image = self._convert_to_numpy(image)
                
                # Perform OCR
                ocr_result = await self.extract_text_from_image(
                    processed_image,
                    document_id=document.id,
                    page_number=page_num,
                    confidence_threshold=self.confidence_threshold,
                    detect_tables=document.processing_config.enable_table_detection,
                    detect_figures=document.processing_config.enable_figure_detection
                )
                
                results.append(ocr_result)
                self.logger.info(f"Processed page {page_num} with confidence {ocr_result.overall_confidence:.2f}")
                
            except Exception as e:
                self.logger.error(f"Failed to process page {page_num}: {e}")
                # Create a failed result
                failed_result = self._create_failed_result(document.id, page_num, str(e))
                results.append(failed_result)
        
        self.logger.info(f"Completed OCR processing for {len(results)} pages")
        return results
    
    async def _extract_images_from_document(self, document: OCRDocument) -> List[np.ndarray]:
        """Extract images from different document types."""
        images = []
        
        if document.is_pdf_document:
            images = await self._extract_pdf_pages(document)
        elif document.is_image_document:
            if document.file_path:
                image = cv2.imread(document.file_path)
                if image is not None:
                    images.append(image)
            elif document.content:
                from io import BytesIO
                import PIL.Image
                pil_image = PIL.Image.open(BytesIO(document.content))
                images.append(np.array(pil_image))
        
        return images
    
    async def _extract_pdf_pages(self, document: OCRDocument) -> List[np.ndarray]:
        """Extract pages from PDF as images."""
        try:
            import fitz  # PyMuPDF
            from pdf2image import convert_from_path, convert_from_bytes
            
            images = []
            
            if document.file_path:
                # Use pdf2image for better quality
                pil_images = convert_from_path(
                    document.file_path,
                    dpi=300,
                    first_page=1,
                    last_page=None
                )
                images = [np.array(img) for img in pil_images]
            elif document.content:
                pil_images = convert_from_bytes(
                    document.content,
                    dpi=300
                )
                images = [np.array(img) for img in pil_images]
            
            return images
            
        except ImportError:
            self.logger.warning("pdf2image not available, falling back to PyMuPDF")
            return await self._extract_pdf_pages_pymupdf(document)
        except Exception as e:
            self.logger.error(f"Failed to extract PDF pages: {e}")
            return []
    
    async def _extract_pdf_pages_pymupdf(self, document: OCRDocument) -> List[np.ndarray]:
        """Extract PDF pages using PyMuPDF."""
        try:
            import fitz
            
            if document.file_path:
                doc = fitz.open(document.file_path)
            elif document.content:
                doc = fitz.open(stream=document.content, filetype="pdf")
            else:
                return []
            
            images = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Convert to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                
                # Convert to numpy array
                from PIL import Image
                from io import BytesIO
                pil_image = Image.open(BytesIO(img_data))
                images.append(np.array(pil_image))
            
            doc.close()
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to extract PDF pages with PyMuPDF: {e}")
            return []
    
    def _convert_to_numpy(self, image: Union[np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """Convert image to numpy array."""
        return self.preprocessor._convert_to_numpy(image)
    
    def _create_failed_result(self, document_id: str, page_number: int, error_message: str) -> OCRResult:
        """Create a failed OCR result."""
        return OCRResult(
            document_id=document_id,
            page_number=page_number,
            regions=[],
            tables=[],
            ocr_engine=OCREngine(self.engine_name),
            processing_time=0.0,
            image_dimensions=(0, 0),
            overall_confidence=0.0,
            text_coverage_ratio=0.0,
            low_confidence_regions=0,
            preprocessing_applied=[],
            confidence_threshold=self.confidence_threshold
        )
    
    def _calculate_overall_confidence(self, regions: List[OCRRegion]) -> float:
        """Calculate overall confidence from regions."""
        if not regions:
            return 0.0
        return sum(region.confidence for region in regions) / len(regions)
    
    def _calculate_text_coverage(self, regions: List[OCRRegion], image_dims: Tuple[int, int]) -> float:
        """Calculate text coverage ratio."""
        if not regions or not any(image_dims):
            return 0.0
        
        total_area = image_dims[0] * image_dims[1]
        text_area = sum(region.bounding_box.area for region in regions)
        
        return min(text_area / total_area, 1.0)
    
    def _filter_low_confidence_regions(self, regions: List[OCRRegion]) -> List[OCRRegion]:
        """Filter out regions below confidence threshold."""
        return [region for region in regions if region.confidence >= self.confidence_threshold]
