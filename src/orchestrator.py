"""
Main OCR-RAG Orchestrator that coordinates all components.
"""

import asyncio
import uuid
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import logging

from .models import (
    OCRDocument, ProcessingResult, ProcessingStatus, ProcessingMetrics,
    OCRResult, OCRChunk, QualityAssessment
)
from .ocr_processors import create_ocr_processor, create_ocr_pipeline, OCRPipeline
from .chunkers import create_ocr_chunker
from .quality_assessors import OCRQualityAssessor
from .document_analyzers import LayoutAnalyzer

logger = logging.getLogger(__name__)


class OCRRAGOrchestrator:
    """
    Main orchestrator for OCR-focused RAG processing.
    
    Coordinates OCR processing, quality assessment, chunking, and indexing
    for complex documents requiring advanced OCR capabilities.
    """
    
    def __init__(self,
                 ocr_engines: List[str] = None,
                 chunking_strategy: str = "layout_aware",
                 confidence_threshold: float = 0.7,
                 enable_quality_assessment: bool = True,
                 enable_preprocessing: bool = True,
                 max_concurrent_processing: int = 3,
                 enable_batch_processing: bool = True,
                 **kwargs):
        """
        Initialize OCR-RAG orchestrator.
        
        Args:
            ocr_engines: List of OCR engine names to use
            chunking_strategy: Chunking strategy to use
            confidence_threshold: Minimum OCR confidence threshold
            enable_quality_assessment: Enable quality assessment
            enable_preprocessing: Enable image preprocessing
            max_concurrent_processing: Maximum concurrent documents to process
            enable_batch_processing: Enable batch processing capabilities
            **kwargs: Additional configuration parameters
        """
        self.ocr_engines = ocr_engines or ["easyocr", "tesseract"]
        self.chunking_strategy = chunking_strategy
        self.confidence_threshold = confidence_threshold
        self.enable_quality_assessment = enable_quality_assessment
        self.enable_preprocessing = enable_preprocessing
        self.max_concurrent_processing = max_concurrent_processing
        self.enable_batch_processing = enable_batch_processing
        self.config = kwargs
        
        # Initialize components
        self._initialize_components()
        
        # Processing state
        self.active_sessions = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_processing)
        
        logger.info(f"OCR-RAG Orchestrator initialized with engines: {self.ocr_engines}")
    
    def _initialize_components(self):
        """Initialize OCR and processing components."""
        try:
            # Initialize OCR pipeline
            self.ocr_pipeline = create_ocr_pipeline(
                engines=self.ocr_engines,
                confidence_threshold=self.confidence_threshold,
                enable_fallback=True,
                parallel_processing=self.config.get('parallel_ocr', False),
                **self.config
            )
            
            # Initialize chunker
            self.chunker = create_ocr_chunker(
                strategy=self.chunking_strategy,
                chunk_size=self.config.get('chunk_size', 1000),
                chunk_overlap=self.config.get('chunk_overlap', 200),
                confidence_threshold=self.confidence_threshold,
                **self.config
            )
            
            # Initialize quality assessor
            if self.enable_quality_assessment:
                self.quality_assessor = OCRQualityAssessor(
                    min_confidence_threshold=self.confidence_threshold,
                    **self.config
                )
            else:
                self.quality_assessor = None
            
            # Initialize layout analyzer
            self.layout_analyzer = LayoutAnalyzer(**self.config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def process_document(self, document: OCRDocument) -> ProcessingResult:
        """
        Process a single document through the complete OCR-RAG pipeline.
        
        Args:
            document: OCR document to process
            
        Returns:
            Processing result with extracted content and metadata
        """
        session_id = str(uuid.uuid4())
        document_id = document.id or str(uuid.uuid4())
        document.id = document_id
        
        logger.info(f"Starting document processing: {document_id}")
        
        # Create processing result
        result = ProcessingResult(
            document_id=document_id,
            session_id=session_id,
            status=ProcessingStatus.PENDING,
            started_at=datetime.utcnow(),
            metrics=ProcessingMetrics(),
            processing_config=document.processing_config.model_dump()
        )
        
        # Register session
        self.active_sessions[session_id] = result
        
        async with self._semaphore:
            try:
                # Process document through pipeline
                await self._process_document_pipeline(document, result)
                
            except Exception as e:
                result.add_error("processing", e, is_recoverable=False)
                logger.error(f"Document processing failed: {e}")
            finally:
                result.finalize()
                # Clean up session
                self.active_sessions.pop(session_id, None)
        
        logger.info(f"Document processing completed: {document_id}, "
                   f"status: {result.status.value}, "
                   f"chunks: {len(result.chunks)}")
        
        return result
    
    async def _process_document_pipeline(self, document: OCRDocument, result: ProcessingResult):
        """Execute the complete processing pipeline."""
        
        # Step 1: OCR Extraction
        result.status = ProcessingStatus.OCR_EXTRACTION
        ocr_start = datetime.utcnow()
        
        try:
            ocr_results = await self.ocr_pipeline.process_document(document)
            result.ocr_results = ocr_results
            result.metrics.ocr_time = (datetime.utcnow() - ocr_start).total_seconds()
            result.metrics.pages_processed = len(ocr_results)
            
            logger.info(f"OCR extraction completed: {len(ocr_results)} pages")
            
        except Exception as e:
            result.add_error("ocr_extraction", e)
            raise
        
        # Step 2: Quality Assessment
        if self.enable_quality_assessment and self.quality_assessor:
            result.status = ProcessingStatus.QUALITY_ASSESSMENT
            quality_start = datetime.utcnow()
            
            try:
                assessments = []
                for ocr_result in ocr_results:
                    assessment = await self.quality_assessor.assess_ocr_result(ocr_result)
                    assessments.append(assessment)
                
                # Aggregate quality metrics
                await self._aggregate_quality_metrics(assessments, result)
                result.metrics.pages_failed = sum(1 for a in assessments if not a.is_acceptable_for_rag)
                
                quality_time = (datetime.utcnow() - quality_start).total_seconds()
                logger.info(f"Quality assessment completed in {quality_time:.2f}s")
                
            except Exception as e:
                result.add_warning(f"Quality assessment failed: {e}")
        
        # Step 3: Layout Analysis
        result.status = ProcessingStatus.LAYOUT_ANALYSIS
        layout_start = datetime.utcnow()
        
        try:
            # Enhance OCR results with layout analysis
            enhanced_results = []
            for ocr_result in ocr_results:
                enhanced_result = await self.layout_analyzer.analyze_layout(ocr_result)
                enhanced_results.append(enhanced_result)
            
            result.ocr_results = enhanced_results
            layout_time = (datetime.utcnow() - layout_start).total_seconds()
            logger.info(f"Layout analysis completed in {layout_time:.2f}s")
            
        except Exception as e:
            result.add_warning(f"Layout analysis failed: {e}")
            # Continue with original results
        
        # Step 4: Chunking
        result.status = ProcessingStatus.CHUNKING
        chunking_start = datetime.utcnow()
        
        try:
            chunks = self.chunker.chunk_multiple_results(result.ocr_results)
            result.chunks = chunks
            result.metrics.chunking_time = (datetime.utcnow() - chunking_start).total_seconds()
            result.metrics.total_chunks_created = len(chunks)
            result.metrics.high_quality_chunks = len([c for c in chunks if c.is_high_confidence])
            result.metrics.filtered_chunks = len([c for c in chunks if c.is_filtered])
            
            logger.info(f"Chunking completed: {len(chunks)} chunks created")
            
        except Exception as e:
            result.add_error("chunking", e)
            raise
        
        # Step 5: Calculate final metrics
        await self._calculate_final_metrics(result)
        
        result.status = ProcessingStatus.COMPLETED
    
    async def _aggregate_quality_metrics(self, assessments: List[QualityAssessment], result: ProcessingResult):
        """Aggregate quality metrics from multiple page assessments."""
        if not assessments:
            return
        
        # Calculate overall quality score
        quality_scores = [a.overall_quality_score for a in assessments]
        result.overall_quality_score = sum(quality_scores) / len(quality_scores)
        
        # Aggregate confidence metrics
        confidence_means = [a.confidence_metrics.mean for a in assessments]
        result.metrics.average_ocr_confidence = sum(confidence_means) / len(confidence_means)
        result.metrics.min_page_confidence = min(confidence_means)
        result.metrics.max_page_confidence = max(confidence_means)
        
        # Aggregate content metrics
        result.metrics.total_text_extracted = sum(a.content_metrics.total_characters for a in assessments)
        result.metrics.tables_detected = sum(a.content_metrics.tables_detected for a in assessments)
        result.metrics.figures_detected = sum(a.content_metrics.figures_detected for a in assessments)
        
        # Collect detected languages
        all_languages = set()
        for assessment in assessments:
            if assessment.content_metrics.primary_language:
                all_languages.add(assessment.content_metrics.primary_language)
        result.metrics.languages_detected = list(all_languages)
        
        # Create detailed quality assessment
        result.quality_assessment = {
            'overall_score': result.overall_quality_score,
            'page_scores': quality_scores,
            'acceptable_pages': sum(1 for a in assessments if a.is_acceptable_for_rag),
            'total_pages': len(assessments),
            'common_issues': self._find_common_issues(assessments),
            'recommendations': self._generate_recommendations(assessments)
        }
    
    async def _calculate_final_metrics(self, result: ProcessingResult):
        """Calculate final processing metrics."""
        if result.chunks:
            # Text extraction metrics
            total_chars = sum(len(chunk.content) for chunk in result.chunks)
            result.metrics.total_text_extracted = max(result.metrics.total_text_extracted, total_chars)
        
        # Processing efficiency metrics
        if result.started_at and result.completed_at:
            total_time = (result.completed_at - result.started_at).total_seconds()
            result.metrics.total_processing_time = total_time
    
    def _find_common_issues(self, assessments: List[QualityAssessment]) -> List[str]:
        """Find common quality issues across assessments."""
        issue_counts = {}
        
        for assessment in assessments:
            for issue in assessment.detected_issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1
        
        # Return issues that appear in more than 25% of pages
        threshold = len(assessments) * 0.25
        common_issues = [issue for issue, count in issue_counts.items() if count > threshold]
        
        return common_issues
    
    def _generate_recommendations(self, assessments: List[QualityAssessment]) -> List[str]:
        """Generate processing recommendations based on assessments."""
        recommendations = set()
        
        for assessment in assessments:
            recommendations.update(assessment.recommendations)
        
        return list(recommendations)
    
    async def batch_process_documents(self, documents: List[OCRDocument]) -> List[ProcessingResult]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processing results
        """
        if not self.enable_batch_processing:
            raise ValueError("Batch processing is disabled")
        
        logger.info(f"Starting batch processing of {len(documents)} documents")
        
        # Create tasks for concurrent processing
        tasks = []
        for document in documents:
            task = asyncio.create_task(self.process_document(document))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Document {i} failed: {result}")
                # Create failed result
                failed_result = ProcessingResult(
                    document_id=documents[i].id or f"failed_{i}",
                    status=ProcessingStatus.FAILED,
                    started_at=datetime.utcnow(),
                    metrics=ProcessingMetrics()
                )
                failed_result.add_error("batch_processing", result, is_recoverable=False)
                failed_result.finalize()
                processed_results.append(failed_result)
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.is_successful)
        logger.info(f"Batch processing completed: {successful}/{len(documents)} successful")
        
        return processed_results
    
    async def get_processing_status(self, session_id: str) -> Optional[ProcessingResult]:
        """Get current processing status for a session."""
        return self.active_sessions.get(session_id)
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get overall processing metrics and statistics."""
        active_sessions = len(self.active_sessions)
        available_engines = self.ocr_pipeline.get_available_engines()
        engine_status = self.ocr_pipeline.get_engine_status()
        
        return {
            'active_sessions': active_sessions,
            'available_ocr_engines': available_engines,
            'engine_status': engine_status,
            'max_concurrent_processing': self.max_concurrent_processing,
            'chunking_strategy': self.chunking_strategy,
            'confidence_threshold': self.confidence_threshold,
            'quality_assessment_enabled': self.enable_quality_assessment,
            'preprocessing_enabled': self.enable_preprocessing,
            'batch_processing_enabled': self.enable_batch_processing
        }
    
    def configure(self, **kwargs):
        """Update orchestrator configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated configuration: {key} = {value}")
            else:
                self.config[key] = value
        
        # Reinitialize components if needed
        component_configs = {
            'ocr_engines', 'chunking_strategy', 'confidence_threshold',
            'enable_quality_assessment', 'enable_preprocessing'
        }
        
        if any(key in component_configs for key in kwargs.keys()):
            logger.info("Configuration changed, reinitializing components...")
            self._initialize_components()
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator."""
        logger.info("Shutting down OCR-RAG orchestrator...")
        
        # Wait for active sessions to complete
        if self.active_sessions:
            logger.info(f"Waiting for {len(self.active_sessions)} active sessions to complete...")
            while self.active_sessions:
                await asyncio.sleep(1)
        
        logger.info("OCR-RAG orchestrator shutdown complete")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return ['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']
    
    def get_chunking_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        return [
            'layout_aware',
            'confidence_based',
            'hybrid_semantic_layout',
            'table_preserving',
            'fixed_size'
        ]
