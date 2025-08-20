"""
Enhanced Usage Example for RAG FlexOCR Orchestrator

This example demonstrates the usage of the enhanced orchestrator with:
- TrOCR and LayoutLMv3 for advanced OCR
- Unstructured.io for document parsing
- LangChain for intelligent chunking
- Redis caching for performance
- Comprehensive logging throughout
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.enhanced_orchestrator import (
    EnhancedOCROrchestrator,
    EnhancedOCRConfig,
    process_document_with_best_practices
)
from src.models import OCRDocument

# Configure logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def example_basic_processing():
    """
    Basic example of processing a single document with enhanced features.
    """
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Basic Document Processing")
    logger.info("="*60)
    
    # Configure the enhanced orchestrator
    config = EnhancedOCRConfig(
        # Enable advanced OCR models
        ocr_engines=["trocr", "layoutlmv3", "tesseract"],
        enable_trocr=True,
        enable_layoutlmv3=True,
        
        # Use Unstructured.io for parsing
        use_unstructured_io=True,
        
        # Enable LangChain chunking
        use_langchain_splitters=True,
        chunking_strategies=["semantic", "recursive"],
        chunk_size=1000,
        chunk_overlap=200,
        
        # Enable caching
        enable_redis_cache=True,
        cache_ttl_hours=24,
        
        # Enable monitoring
        enable_prometheus=True,
        prometheus_port=8000,
        
        # Detailed logging
        enable_detailed_logging=True,
        log_level="INFO"
    )
    
    # Create orchestrator
    orchestrator = EnhancedOCROrchestrator(config)
    
    # Process a document
    document = OCRDocument(
        file_path="sample_documents/complex_layout.pdf",
        metadata={"source": "example", "type": "pdf"}
    )
    
    try:
        result = await orchestrator.process_document_enhanced(document)
        
        logger.info("\n" + "-"*40)
        logger.info("Processing Results:")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Pages Processed: {result.pages_processed}")
        logger.info(f"  Chunks Created: {result.chunks_created}")
        logger.info(f"  Average Confidence: {result.avg_confidence:.3f}")
        logger.info(f"  Processing Time: {result.processing_time:.2f}s")
        
        if result.metadata:
            logger.info(f"  Models Used: {result.metadata.get('models_used', [])}")
            logger.info(f"  Chunking Strategy: {result.metadata.get('chunking_strategy', 'N/A')}")
        
        if result.errors:
            logger.error(f"  Errors: {result.errors}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)


async def example_trocr_processing():
    """
    Example specifically demonstrating TrOCR for high-quality OCR.
    """
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: TrOCR Advanced OCR")
    logger.info("="*60)
    
    from src.ocr_processors.trocr_processor import TrOCRProcessor
    from PIL import Image
    
    # Initialize TrOCR processor
    trocr = TrOCRProcessor(
        model_name="printed_large",  # Use large model for best quality
        enable_preprocessing=True,
        enable_beam_search=True,
        num_beams=5,
        batch_size=4
    )
    
    logger.info(f"TrOCR Processor: {trocr}")
    
    # Process an image
    image_path = "sample_documents/scanned_page.png"
    
    try:
        # Load image
        image = Image.open(image_path)
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Image size: {image.size}, mode: {image.mode}")
        
        # Extract text with TrOCR
        result = await trocr.extract_text_from_image(image)
        
        logger.info("\n" + "-"*40)
        logger.info("TrOCR Results:")
        logger.info(f"  Extracted Text Length: {len(result.extracted_text)}")
        logger.info(f"  Confidence: {result.confidence_scores[0] if result.confidence_scores else 0:.3f}")
        logger.info(f"  Processing Time: {result.processing_time_seconds:.2f}s")
        logger.info(f"\nExtracted Text (first 500 chars):")
        logger.info(f"  {result.extracted_text[:500]}...")
        
    except Exception as e:
        logger.error(f"TrOCR processing failed: {e}", exc_info=True)


async def example_layoutlmv3_processing():
    """
    Example demonstrating LayoutLMv3 for document understanding.
    """
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: LayoutLMv3 Document Understanding")
    logger.info("="*60)
    
    from src.ocr_processors.layoutlmv3_processor import LayoutLMv3Processor
    from PIL import Image
    
    # Initialize LayoutLMv3 processor
    layoutlm = LayoutLMv3Processor(
        model_name="base",
        enable_ocr=True,
        enable_visual_features=True,
        batch_size=2
    )
    
    logger.info(f"LayoutLMv3 Processor: {layoutlm}")
    
    # Process a document with complex layout
    image_path = "sample_documents/form_document.png"
    
    try:
        # Load image
        image = Image.open(image_path)
        logger.info(f"Processing document: {image_path}")
        
        # Extract text with layout understanding
        result = await layoutlm.extract_text_from_image(image)
        
        logger.info("\n" + "-"*40)
        logger.info("LayoutLMv3 Results:")
        logger.info(f"  Regions Detected: {len(result.regions)}")
        logger.info(f"  Processing Time: {result.processing_time_seconds:.2f}s")
        
        # Show detected regions by type
        region_types = {}
        for region in result.regions:
            region_type = region.region_type.value
            region_types[region_type] = region_types.get(region_type, 0) + 1
        
        logger.info("  Region Types Detected:")
        for region_type, count in region_types.items():
            logger.info(f"    {region_type}: {count}")
        
        # Extract tables if any
        tables = await layoutlm.extract_tables(image)
        if tables:
            logger.info(f"\n  Tables Found: {len(tables)}")
            for i, table in enumerate(tables):
                logger.info(f"    Table {i+1}: {table.get('confidence', 0):.3f} confidence")
        
    except Exception as e:
        logger.error(f"LayoutLMv3 processing failed: {e}", exc_info=True)


async def example_batch_processing():
    """
    Example of batch processing multiple documents.
    """
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Batch Document Processing")
    logger.info("="*60)
    
    config = EnhancedOCRConfig(
        ocr_engines=["trocr", "layoutlmv3"],
        enable_trocr=True,
        enable_layoutlmv3=True,
        use_unstructured_io=True,
        use_langchain_splitters=True,
        enable_redis_cache=True,
        max_workers=4,
        batch_size=5
    )
    
    orchestrator = EnhancedOCROrchestrator(config)
    
    # Create multiple documents
    documents = [
        OCRDocument(file_path=f"sample_documents/doc_{i}.pdf")
        for i in range(1, 4)
    ]
    
    logger.info(f"Processing {len(documents)} documents in batch")
    
    try:
        results = await orchestrator.batch_process_enhanced(
            documents,
            use_ray=False  # Set to True if Ray is installed
        )
        
        logger.info("\n" + "-"*40)
        logger.info("Batch Processing Results:")
        
        successful = sum(1 for r in results if r.status.value == "completed")
        failed = len(results) - successful
        
        logger.info(f"  Total Documents: {len(documents)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        
        total_chunks = sum(r.chunks_created for r in results)
        avg_confidence = sum(r.avg_confidence for r in results) / len(results)
        total_time = sum(r.processing_time for r in results)
        
        logger.info(f"  Total Chunks Created: {total_chunks}")
        logger.info(f"  Average Confidence: {avg_confidence:.3f}")
        logger.info(f"  Total Processing Time: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)


async def example_quality_assessment():
    """
    Example demonstrating advanced quality assessment.
    """
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Advanced Quality Assessment")
    logger.info("="*60)
    
    orchestrator = EnhancedOCROrchestrator(
        EnhancedOCRConfig(
            use_textstat_metrics=True,
            enable_detailed_logging=True
        )
    )
    
    # Sample text for quality assessment
    sample_text = """
    This is a sample text for quality assessment. It contains multiple sentences
    with varying complexity levels. The quality assessment will analyze readability,
    grammar, entity density, and other linguistic features.
    
    Some technical terms like machine learning, neural networks, and transformers
    are included. This helps test entity recognition capabilities.
    
    The text also includes some intentional issues for grammar checking.
    """
    
    try:
        quality_metrics = await orchestrator.assess_quality_with_advanced_metrics(sample_text)
        
        logger.info("\n" + "-"*40)
        logger.info("Quality Assessment Results:")
        
        for metric, value in quality_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.3f}")
            else:
                logger.info(f"  {metric}: {value}")
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}", exc_info=True)


async def example_with_caching():
    """
    Example demonstrating caching benefits.
    """
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 6: Performance with Caching")
    logger.info("="*60)
    
    config = EnhancedOCRConfig(
        enable_redis_cache=True,
        cache_ttl_hours=1,
        enable_detailed_logging=True
    )
    
    orchestrator = EnhancedOCROrchestrator(config)
    
    document = OCRDocument(file_path="sample_documents/test.pdf")
    
    try:
        # First processing (cache miss)
        logger.info("First processing (cache miss expected)...")
        result1 = await orchestrator.process_document_enhanced(document)
        time1 = result1.processing_time
        logger.info(f"  Processing time: {time1:.2f}s")
        
        # Second processing (cache hit)
        logger.info("\nSecond processing (cache hit expected)...")
        result2 = await orchestrator.process_document_enhanced(document)
        time2 = result2.processing_time
        logger.info(f"  Processing time: {time2:.2f}s")
        
        if time2 < time1:
            speedup = (time1 - time2) / time1 * 100
            logger.info(f"\n✓ Caching provided {speedup:.1f}% speedup!")
        
    except Exception as e:
        logger.error(f"Caching example failed: {e}", exc_info=True)


async def main():
    """
    Run all examples.
    """
    logger.info("\n" + "="*80)
    logger.info("RAG FlexOCR Orchestrator - Enhanced Examples")
    logger.info("="*80)
    
    # Create sample documents directory if it doesn't exist
    sample_dir = Path("sample_documents")
    if not sample_dir.exists():
        logger.warning(f"Sample documents directory not found: {sample_dir}")
        logger.info("Creating sample documents directory...")
        sample_dir.mkdir(exist_ok=True)
        
        # Create a simple test file
        test_file = sample_dir / "test.txt"
        test_file.write_text("This is a test document for the enhanced OCR orchestrator.")
        logger.info(f"Created test file: {test_file}")
    
    # Run examples
    examples = [
        # example_basic_processing,
        # example_trocr_processing,
        # example_layoutlmv3_processing,
        # example_batch_processing,
        example_quality_assessment,
        # example_with_caching,
    ]
    
    for example_func in examples:
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example {example_func.__name__} failed: {e}")
    
    logger.info("\n" + "="*80)
    logger.info("All examples completed!")
    logger.info("="*80)
    
    # Show monitoring information if Prometheus is enabled
    logger.info("\n📊 Monitoring Information:")
    logger.info("If Prometheus is enabled, metrics are available at:")
    logger.info("  http://localhost:8000/metrics")
    logger.info("\nKey metrics include:")
    logger.info("  - ocr_documents_processed: Total documents processed")
    logger.info("  - ocr_processing_duration_seconds: Processing time histogram")
    logger.info("  - ocr_quality_score: Current quality score gauge")
    logger.info("  - cache_hits/cache_misses: Cache performance")
    logger.info("  - ocr_processing_errors: Error counts by type")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())

