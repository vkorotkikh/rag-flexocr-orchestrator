"""
Factory methods for creating OCR processors and processing pipelines.
"""

from typing import List, Dict, Any, Optional, Union
import logging

from .base import BaseOCRProcessor
from .tesseract_processor import TesseractProcessor
from .easyocr_processor import EasyOCRProcessor
from .aws_textract_processor import AWSTextractProcessor

logger = logging.getLogger(__name__)


def create_ocr_processor(
    engine: str,
    confidence_threshold: float = 0.7,
    enable_preprocessing: bool = True,
    **kwargs
) -> BaseOCRProcessor:
    """
    Create an OCR processor instance.
    
    Args:
        engine: OCR engine name ('tesseract', 'easyocr', 'aws_textract')
        confidence_threshold: Minimum confidence threshold
        enable_preprocessing: Whether to enable image preprocessing
        **kwargs: Engine-specific configuration parameters
        
    Returns:
        OCR processor instance
        
    Raises:
        ValueError: If engine is not supported
        RuntimeError: If engine is not available
    """
    engine = engine.lower().strip()
    
    if engine == 'tesseract':
        processor = TesseractProcessor(
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing,
            language=kwargs.get('language', 'eng'),
            page_segmentation_mode=kwargs.get('psm', 6),
            ocr_engine_mode=kwargs.get('oem', 3)
        )
    
    elif engine == 'easyocr':
        processor = EasyOCRProcessor(
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing,
            languages=kwargs.get('languages', ['en']),
            use_gpu=kwargs.get('use_gpu', False),
            detector_backend=kwargs.get('detector_backend', 'craft'),
            recognizer_backend=kwargs.get('recognizer_backend', 'crnn')
        )
    
    elif engine == 'aws_textract':
        processor = AWSTextractProcessor(
            confidence_threshold=confidence_threshold,
            enable_preprocessing=enable_preprocessing,
            aws_region=kwargs.get('aws_region', 'us-east-1'),
            enable_tables=kwargs.get('enable_tables', True),
            enable_forms=kwargs.get('enable_forms', True)
        )
    
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}")
    
    # Verify processor is available
    if not processor.is_available():
        raise RuntimeError(f"OCR engine {engine} is not available")
    
    logger.info(f"Created {engine} OCR processor")
    return processor


class OCRPipeline:
    """
    OCR processing pipeline with multiple engines and fallback support.
    """
    
    def __init__(self, 
                 engine_configs: List[Dict[str, Any]],
                 enable_fallback: bool = True,
                 parallel_processing: bool = False):
        """
        Initialize OCR pipeline.
        
        Args:
            engine_configs: List of engine configurations
            enable_fallback: Enable fallback to next engine on failure
            parallel_processing: Process with multiple engines in parallel
        """
        self.engine_configs = engine_configs
        self.enable_fallback = enable_fallback
        self.parallel_processing = parallel_processing
        
        # Initialize processors
        self.processors = []
        self.available_processors = []
        
        for config in engine_configs:
            try:
                processor = create_ocr_processor(**config)
                self.processors.append(processor)
                if processor.is_available():
                    self.available_processors.append(processor)
                    logger.info(f"Added {processor.engine_name} to pipeline")
            except Exception as e:
                logger.warning(f"Failed to initialize processor {config.get('engine')}: {e}")
        
        if not self.available_processors:
            raise RuntimeError("No OCR processors are available")
    
    async def process_document(self, document) -> List:
        """Process document with the OCR pipeline."""
        if not self.available_processors:
            raise RuntimeError("No available OCR processors")
        
        if self.parallel_processing:
            return await self._process_parallel(document)
        else:
            return await self._process_sequential(document)
    
    async def _process_sequential(self, document) -> List:
        """Process document sequentially with fallback."""
        last_error = None
        
        for processor in self.available_processors:
            try:
                logger.info(f"Processing with {processor.engine_name}")
                results = await processor.process_document(document)
                
                # Check if results are acceptable
                if self._validate_results(results):
                    logger.info(f"Successfully processed with {processor.engine_name}")
                    return results
                else:
                    logger.warning(f"Poor quality results from {processor.engine_name}")
                    if not self.enable_fallback:
                        return results
                    
            except Exception as e:
                logger.error(f"Processing failed with {processor.engine_name}: {e}")
                last_error = e
                if not self.enable_fallback:
                    raise
        
        # If we get here, all processors failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError("All OCR processors failed")
    
    async def _process_parallel(self, document) -> List:
        """Process document with multiple engines in parallel."""
        import asyncio
        
        # Create tasks for all available processors
        tasks = []
        for processor in self.available_processors:
            task = asyncio.create_task(
                self._safe_process(processor, document)
            )
            tasks.append((processor.engine_name, task))
        
        # Wait for all tasks to complete
        results = {}
        for engine_name, task in tasks:
            try:
                result = await task
                if result and self._validate_results(result):
                    results[engine_name] = result
            except Exception as e:
                logger.error(f"Parallel processing failed for {engine_name}: {e}")
        
        if not results:
            raise RuntimeError("All OCR processors failed in parallel mode")
        
        # Return the best result (highest average confidence)
        best_engine = max(results.keys(), 
                         key=lambda k: self._calculate_average_confidence(results[k]))
        
        logger.info(f"Best results from {best_engine} in parallel processing")
        return results[best_engine]
    
    async def _safe_process(self, processor, document):
        """Safely process document with error handling."""
        try:
            return await processor.process_document(document)
        except Exception as e:
            logger.warning(f"Safe processing failed for {processor.engine_name}: {e}")
            return None
    
    def _validate_results(self, results: List) -> bool:
        """Validate OCR results quality."""
        if not results:
            return False
        
        # Check if we have reasonable confidence and text coverage
        total_confidence = 0
        total_coverage = 0
        valid_results = 0
        
        for result in results:
            if result.overall_confidence > 0:
                total_confidence += result.overall_confidence
                total_coverage += result.text_coverage_ratio
                valid_results += 1
        
        if valid_results == 0:
            return False
        
        avg_confidence = total_confidence / valid_results
        avg_coverage = total_coverage / valid_results
        
        # Minimum thresholds for acceptable results
        return avg_confidence >= 0.6 and avg_coverage >= 0.05
    
    def _calculate_average_confidence(self, results: List) -> float:
        """Calculate average confidence across all results."""
        if not results:
            return 0.0
        
        total_confidence = sum(result.overall_confidence for result in results)
        return total_confidence / len(results)
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        return [processor.engine_name for processor in self.available_processors]
    
    def get_engine_status(self) -> Dict[str, bool]:
        """Get availability status of all configured engines."""
        status = {}
        for processor in self.processors:
            status[processor.engine_name] = processor.is_available()
        return status


def create_ocr_pipeline(
    engines: Union[str, List[str], List[Dict[str, Any]]],
    confidence_threshold: float = 0.7,
    enable_fallback: bool = True,
    parallel_processing: bool = False,
    **default_kwargs
) -> OCRPipeline:
    """
    Create an OCR processing pipeline.
    
    Args:
        engines: Engine name(s) or configurations
        confidence_threshold: Default confidence threshold
        enable_fallback: Enable fallback between engines
        parallel_processing: Enable parallel processing
        **default_kwargs: Default parameters for all engines
        
    Returns:
        OCR pipeline instance
    """
    # Normalize engines parameter
    if isinstance(engines, str):
        engines = [engines]
    
    # Convert to configurations
    engine_configs = []
    for engine in engines:
        if isinstance(engine, dict):
            # Already a configuration
            config = engine.copy()
        else:
            # Engine name - create default configuration
            config = {
                'engine': engine,
                'confidence_threshold': confidence_threshold,
                **default_kwargs
            }
        
        engine_configs.append(config)
    
    return OCRPipeline(
        engine_configs=engine_configs,
        enable_fallback=enable_fallback,
        parallel_processing=parallel_processing
    )


def get_available_engines() -> List[str]:
    """Get list of all available OCR engines on the system."""
    available = []
    
    engines_to_test = [
        ('tesseract', TesseractProcessor),
        ('easyocr', EasyOCRProcessor),
        ('aws_textract', AWSTextractProcessor)
    ]
    
    for engine_name, processor_class in engines_to_test:
        try:
            processor = processor_class()
            if processor.is_available():
                available.append(engine_name)
        except Exception:
            pass  # Engine not available
    
    return available


def create_optimal_pipeline(
    document_type: str = 'general',
    quality_priority: str = 'balanced',  # 'speed', 'balanced', 'quality'
    enable_cloud_ocr: bool = False,
    **kwargs
) -> OCRPipeline:
    """
    Create an optimized OCR pipeline based on document type and priorities.
    
    Args:
        document_type: Type of documents ('general', 'forms', 'tables', 'handwritten')
        quality_priority: Processing priority ('speed', 'balanced', 'quality')
        enable_cloud_ocr: Whether to include cloud OCR services
        **kwargs: Additional configuration
        
    Returns:
        Optimized OCR pipeline
    """
    # Define engine priorities based on document type and quality preference
    engine_priorities = {
        'speed': ['easyocr', 'tesseract'],
        'balanced': ['easyocr', 'tesseract', 'aws_textract'],
        'quality': ['aws_textract', 'easyocr', 'tesseract']
    }
    
    # Document-specific configurations
    document_configs = {
        'forms': {
            'enable_structure_analysis': True,
            'enable_tables': True,
            'enable_forms': True,
            'psm': 6  # For Tesseract
        },
        'tables': {
            'enable_structure_analysis': True,
            'enable_tables': True,
            'psm': 6
        },
        'handwritten': {
            'psm': 8,  # Single word mode for handwritten
            'language': 'eng'
        },
        'general': {
            'enable_structure_analysis': True,
            'psm': 6
        }
    }
    
    # Get engines to use
    engines = engine_priorities.get(quality_priority, ['easyocr', 'tesseract'])
    
    if not enable_cloud_ocr:
        engines = [e for e in engines if e != 'aws_textract']
    
    # Get document-specific config
    doc_config = document_configs.get(document_type, document_configs['general'])
    
    # Merge with provided kwargs
    config = {**doc_config, **kwargs}
    
    # Create engine configurations
    engine_configs = []
    for engine in engines:
        engine_config = {
            'engine': engine,
            'confidence_threshold': config.get('confidence_threshold', 0.7),
            **config
        }
        engine_configs.append(engine_config)
    
    return OCRPipeline(
        engine_configs=engine_configs,
        enable_fallback=True,
        parallel_processing=(quality_priority == 'quality')
    )
