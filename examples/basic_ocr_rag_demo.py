#!/usr/bin/env python3
"""
Basic OCR-RAG demonstration script.

This script demonstrates how to use the OCR-focused RAG system to process
challenging documents and extract meaningful information.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.models import OCRDocument, ProcessingConfig
    from src.ocr_processors import create_ocr_processor, create_ocr_pipeline
    from src.chunkers import create_ocr_chunker
    from src.quality_assessors import OCRQualityAssessor
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("This is a demonstration with simulated imports")
    # Create mock classes for demo
    class OCRDocument:
        def __init__(self, **kwargs):
            self.processing_config = kwargs.get('processing_config', type('Config', (), {})())
            self.document_type = type('Type', (), {'value': 'pdf'})()
    
    class ProcessingConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def create_ocr_processor(**kwargs):
        return type('Processor', (), {'engine_name': kwargs.get('engine', 'mock')})()
    
    def create_ocr_pipeline(**kwargs):
        return None
    
    def create_ocr_chunker(**kwargs):
        return None
    
    class OCRQualityAssessor:
        async def assess_ocr_result(self, result):
            return type('Assessment', (), {
                'overall_quality_score': 0.85,
                'overall_quality_level': type('Level', (), {'value': 'good'})(),
                'is_acceptable_for_rag': True,
                'needs_reprocessing': False,
                'detected_issues': [],
                'recommendations': []
            })()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_ocr_processing():
    """Demonstrate basic OCR processing with different engines."""
    print("🔍 OCR Engine Demonstration")
    print("=" * 50)
    
    # Create a simple test image (would normally be a real document)
    print("\n📝 Testing OCR Engines")
    
    # Test Tesseract
    try:
        tesseract_processor = create_ocr_processor(
            engine='tesseract',
            confidence_threshold=0.7,
            language='eng'
        )
        print("✅ Tesseract OCR processor initialized")
    except Exception as e:
        print(f"❌ Tesseract not available: {e}")
    
    # Test EasyOCR
    try:
        easy_processor = create_ocr_processor(
            engine='easyocr',
            confidence_threshold=0.7,
            languages=['en'],
            use_gpu=False
        )
        print("✅ EasyOCR processor initialized")
    except Exception as e:
        print(f"❌ EasyOCR not available: {e}")
    
    # Test AWS Textract (requires credentials)
    try:
        aws_processor = create_ocr_processor(
            engine='aws_textract',
            confidence_threshold=0.7,
            enable_tables=True
        )
        print("✅ AWS Textract processor initialized")
    except Exception as e:
        print(f"❌ AWS Textract not available: {e}")


async def demo_ocr_pipeline():
    """Demonstrate OCR pipeline with fallback support."""
    print("\n🔄 OCR Pipeline with Fallback")
    print("-" * 35)
    
    # Create pipeline with multiple engines
    pipeline_config = [
        {
            'engine': 'easyocr',
            'confidence_threshold': 0.8,
            'languages': ['en'],
            'use_gpu': False
        },
        {
            'engine': 'tesseract',
            'confidence_threshold': 0.7,
            'language': 'eng',
            'psm': 6
        }
    ]
    
    try:
        from src.ocr_processors.factory import OCRPipeline
        pipeline = OCRPipeline(
            engine_configs=pipeline_config,
            enable_fallback=True,
            parallel_processing=False
        )
        
        available_engines = pipeline.get_available_engines()
        print(f"📊 Available engines: {available_engines}")
        print(f"🔧 Pipeline status: {pipeline.get_engine_status()}")
        
    except Exception as e:
        print(f"❌ Failed to create OCR pipeline: {e}")


async def demo_chunking_strategies():
    """Demonstrate different chunking strategies for OCR content."""
    print("\n✂️  OCR Chunking Strategies")
    print("-" * 30)
    
    # Create sample OCR regions (would normally come from real OCR)
    from src.models import OCRResult, OCRRegion, BoundingBox, RegionType, OCREngine
    
    sample_regions = [
        OCRRegion(
            text="Document Title: Advanced OCR Processing",
            confidence=0.95,
            bounding_box=BoundingBox(x1=50, y1=50, x2=400, y2=80),
            region_type=RegionType.TITLE,
            region_id="title_1"
        ),
        OCRRegion(
            text="This is the first paragraph of our document. It contains important information about OCR processing techniques and how they can be applied to extract meaningful content from challenging documents.",
            confidence=0.88,
            bounding_box=BoundingBox(x1=50, y1=100, x2=400, y2=140),
            region_type=RegionType.PARAGRAPH,
            region_id="para_1"
        ),
        OCRRegion(
            text="• First key point about OCR accuracy",
            confidence=0.82,
            bounding_box=BoundingBox(x1=70, y1=160, x2=380, y2=180),
            region_type=RegionType.LIST,
            region_id="list_1"
        ),
        OCRRegion(
            text="• Second key point about document structure",
            confidence=0.85,
            bounding_box=BoundingBox(x1=70, y1=190, x2=380, y2=210),
            region_type=RegionType.LIST,
            region_id="list_2"
        )
    ]
    
    # Create sample OCR result
    ocr_result = OCRResult(
        document_id="demo_doc",
        page_number=1,
        regions=sample_regions,
                    tables=[],
            ocr_engine=OCREngine.EASYOCR,
            processing_time=1.5,
        image_dimensions=(500, 700),
        overall_confidence=0.87,
        text_coverage_ratio=0.25,
        low_confidence_regions=0,
        confidence_threshold=0.7
    )
    
    # Test Layout-Aware Chunking
    print("\n📋 Layout-Aware Chunking:")
    try:
        from src.chunkers import LayoutAwareChunker
        layout_chunker = LayoutAwareChunker(
            chunk_size=500,
            preserve_headers=True,
            preserve_lists=True
        )
        
        layout_chunks = layout_chunker.chunk_ocr_result(ocr_result)
        print(f"   Created {len(layout_chunks)} layout-aware chunks")
        
        for i, chunk in enumerate(layout_chunks):
            print(f"   Chunk {i+1}: {len(chunk.content)} chars, "
                  f"confidence: {chunk.metadata.average_confidence:.2f}, "
                  f"types: {set(rt.value for rt in chunk.metadata.region_types)}")
    
    except Exception as e:
        print(f"   ❌ Layout chunking failed: {e}")
    
    # Test Confidence-Based Chunking
    print("\n📊 Confidence-Based Chunking:")
    try:
        from src.chunkers import ConfidenceBasedChunker
        confidence_chunker = ConfidenceBasedChunker(
            chunk_size=500,
            high_confidence_threshold=0.9,
            medium_confidence_threshold=0.8,
            separate_confidence_levels=True
        )
        
        confidence_chunks = confidence_chunker.chunk_ocr_result(ocr_result)
        print(f"   Created {len(confidence_chunks)} confidence-based chunks")
        
        for i, chunk in enumerate(confidence_chunks):
            conf_category = "high" if chunk.metadata.average_confidence >= 0.9 else \
                           "medium" if chunk.metadata.average_confidence >= 0.8 else "low"
            print(f"   Chunk {i+1}: {len(chunk.content)} chars, "
                  f"confidence: {chunk.metadata.average_confidence:.2f} ({conf_category})")
    
    except Exception as e:
        print(f"   ❌ Confidence chunking failed: {e}")


async def demo_quality_assessment():
    """Demonstrate OCR quality assessment."""
    print("\n🔍 OCR Quality Assessment")
    print("-" * 28)
    
    # Create sample OCR result with varying quality
    from src.models import OCRResult, OCRRegion, BoundingBox, RegionType, OCREngine
    
    mixed_quality_regions = [
        OCRRegion(
            text="High quality text with excellent OCR confidence",
            confidence=0.95,
            bounding_box=BoundingBox(x1=50, y1=50, x2=400, y2=80),
            region_type=RegionType.PARAGRAPH,
            region_id="high_q_1"
        ),
        OCRRegion(
            text="M3dium qu4lity t3xt w1th s0me 0CR err0rs",
            confidence=0.65,
            bounding_box=BoundingBox(x1=50, y1=100, x2=400, y2=120),
            region_type=RegionType.PARAGRAPH,
            region_id="med_q_1"
        ),
        OCRRegion(
            text="L0w qu@l1ty t3xt w1th m@ny 3rr0rs @nd n01s3",
            confidence=0.35,
            bounding_box=BoundingBox(x1=50, y1=140, x2=400, y2=160),
            region_type=RegionType.PARAGRAPH,
            region_id="low_q_1"
        )
    ]
    
    ocr_result = OCRResult(
        document_id="quality_demo",
        page_number=1,
        regions=mixed_quality_regions,
        tables=[],
        ocr_engine=OCREngine.TESSERACT,
        processing_time=2.1,
        image_dimensions=(500, 300),
        overall_confidence=0.65,
        text_coverage_ratio=0.18,
        low_confidence_regions=1,
        confidence_threshold=0.6
    )
    
    try:
        quality_assessor = OCRQualityAssessor()
        assessment = await quality_assessor.assess_ocr_result(ocr_result)
        
        print(f"📊 Overall Quality Score: {assessment.overall_quality_score:.2f}")
        print(f"🏷️  Quality Level: {assessment.overall_quality_level.value}")
        print(f"✅ Suitable for RAG: {assessment.is_acceptable_for_rag}")
        print(f"🔄 Needs Reprocessing: {assessment.needs_reprocessing}")
        
        if assessment.detected_issues:
            print(f"\n⚠️  Issues Detected:")
            for issue in assessment.detected_issues:
                description = assessment.issue_descriptions.get(issue, "No description")
                print(f"   • {issue.value}: {description}")
        
        if assessment.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in assessment.recommendations:
                print(f"   • {rec}")
    
    except Exception as e:
        print(f"❌ Quality assessment failed: {e}")


async def demo_document_processing():
    """Demonstrate end-to-end document processing."""
    print("\n📄 End-to-End Document Processing")
    print("-" * 38)
    
    # Create a mock document (in practice, this would be a real PDF or image)
    processing_config = ProcessingConfig(
        enhance_quality=True,
        preserve_layout=True,
        detect_language=True,
        confidence_threshold=0.7,
        ocr_engines=["easyocr", "tesseract"],
        preprocessing_steps=["noise_reduction", "contrast_enhancement"],
        enable_table_detection=True
    )
    
    document = OCRDocument(
        content=None,  # Would be real document bytes
        processing_config=processing_config
    )
    
    print(f"📋 Document Type: {document.document_type.value}")
    print(f"🔧 OCR Engines: {document.processing_config.ocr_engines}")
    print(f"⚙️  Preprocessing: {document.processing_config.preprocessing_steps}")
    print(f"📊 Confidence Threshold: {document.processing_config.confidence_threshold}")
    print(f"📋 Table Detection: {document.processing_config.enable_table_detection}")
    
    # Demonstrate processing configuration
    print(f"\n🔍 Processing Features:")
    print(f"   • Quality Enhancement: {document.processing_config.enhance_quality}")
    print(f"   • Layout Preservation: {document.processing_config.preserve_layout}")
    print(f"   • Language Detection: {document.processing_config.detect_language}")
    print(f"   • Max Resolution: {document.processing_config.max_resolution}")


async def demo_performance_comparison():
    """Demonstrate performance comparison between different strategies."""
    print("\n⚡ Performance Comparison")
    print("-" * 26)
    
    strategies = [
        ("Layout-Aware", "Preserves document structure"),
        ("Confidence-Based", "Groups by OCR quality"),
        ("Hybrid Semantic", "Combines meaning and layout"),
        ("Table-Preserving", "Specialized for tables")
    ]
    
    print("📊 Chunking Strategy Comparison:")
    print()
    print("Strategy            | Focus Area              | Best Use Case")
    print("-" * 70)
    
    for strategy, description in strategies:
        use_case = {
            "Layout-Aware": "Complex formatted documents",
            "Confidence-Based": "Variable quality scans",
            "Hybrid Semantic": "Research papers, articles",
            "Table-Preserving": "Financial, technical docs"
        }.get(strategy, "General purpose")
        
        print(f"{strategy:<18} | {description:<22} | {use_case}")
    
    print()
    print("💡 Recommendation: Choose strategy based on:")
    print("   • Document type and complexity")
    print("   • OCR quality and consistency")
    print("   • Required processing accuracy")
    print("   • Performance requirements")


async def main():
    """Main demonstration function."""
    print("🚀 OCR-Focused RAG System Demonstration")
    print("=" * 70)
    print()
    print("This demonstration shows the capabilities of the OCR-focused RAG system")
    print("for processing challenging documents with advanced OCR and chunking.")
    print()
    
    try:
        # Run all demonstrations
        await demo_basic_ocr_processing()
        await demo_ocr_pipeline()
        await demo_chunking_strategies()
        await demo_quality_assessment()
        await demo_document_processing()
        await demo_performance_comparison()
        
        print("\n✅ Demonstration completed successfully!")
        print()
        print("🔗 Next Steps:")
        print("   1. Install required OCR engines (tesseract, easyocr)")
        print("   2. Configure API keys for cloud OCR services")
        print("   3. Test with your own documents")
        print("   4. Integrate with your RAG pipeline")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        logger.exception("Demo failed")


def setup_environment():
    """Setup environment variables for demonstration."""
    print("⚙️  Setting up demonstration environment...")
    
    # Set default values for demo
    env_vars = {
        'LOG_LEVEL': 'INFO',
        'DEFAULT_OCR_ENGINE': 'easyocr',
        'ENABLE_PREPROCESSING': 'true',
        'DEFAULT_CONFIDENCE_THRESHOLD': '0.7'
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print("✅ Environment configured for demonstration")


if __name__ == "__main__":
    print("Setting up demonstration...")
    setup_environment()
    
    print("\n⚠️  DEMONSTRATION NOTES:")
    print("• This demo uses simulated data for illustration")
    print("• Real OCR engines may not be installed")
    print("• Some features require additional dependencies")
    print("• Cloud services require API key configuration")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration error: {e}")
        import traceback
        traceback.print_exc()
