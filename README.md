# RAG FlexOCR Orchestrator

A comprehensive, production-ready RAG system specifically designed for handling unstructured documents that require advanced OCR processing. Built to handle complex PDFs, scanned documents, images, and other challenging document formats with intelligent preprocessing, multi-engine OCR, and layout-aware chunking.

## 🌟 Key Features

### Advanced OCR Processing
- **Multi-Engine OCR**: Tesseract, EasyOCR, AWS Textract with intelligent fallback chains
- **Image Preprocessing**: Noise reduction, contrast enhancement, deskewing, rotation correction
- **Multi-Resolution Processing**: Automatic resolution optimization for best OCR results
- **Language Detection**: Automatic detection and processing of 100+ languages
- **Quality Assessment**: OCR confidence scoring and validation

### Intelligent Document Understanding
- **Layout Detection**: Automatic identification of headers, paragraphs, tables, figures
- **Structure Preservation**: Maintains spatial relationships and document hierarchy
- **Table Extraction**: Specialized handling of tabular data with cell relationships
- **Multi-Column Support**: Smart processing of complex layouts

### Specialized Chunking Strategies
- **Layout-Preserving Chunker**: Respects document structure and formatting
- **Confidence-Based Chunker**: Groups content by OCR reliability scores
- **Hybrid Semantic-Layout**: Combines semantic meaning with layout structure
- **Table-Aware Chunker**: Maintains table integrity and relationships

### Production-Ready Architecture
- **Async Processing**: High-performance async pipeline for batch processing
- **Error Resilience**: Comprehensive error handling and recovery strategies
- **Progress Tracking**: Real-time monitoring for long-running operations
- **Scalable Design**: Designed for enterprise-scale document processing

## 🏗️ Architecture

```
rag-flexocr-orchestrator/
├── src/
│   ├── ocr_processors/        # OCR engines and preprocessing
│   ├── document_analyzers/    # Layout and structure analysis
│   ├── chunkers/             # OCR-aware chunking strategies
│   ├── quality_assessors/    # OCR quality and validation
│   ├── models/               # Pydantic data models
│   ├── utils/                # Utilities and helpers
│   ├── config/               # Configuration management
│   └── orchestrator.py       # Main OCR-RAG pipeline
├── tests/                    # Comprehensive test suite
├── examples/                 # Usage examples and demos
├── docs/                     # Documentation
└── sample_documents/         # Test documents for examples
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd rag-flexocr-orchestrator

# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies for OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-eng
sudo apt-get install poppler-utils

# macOS:
brew install tesseract
brew install poppler
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
```

### Basic Usage

```python
import asyncio
from src.orchestrator import OCRRAGOrchestrator
from src.models import OCRDocument

async def main():
    # Initialize OCR-RAG system
    orchestrator = OCRRAGOrchestrator(
        ocr_engines=["easyocr", "tesseract"],
        chunking_strategy="layout_aware",
        vector_store="pinecone"
    )
    
    # Process a challenging PDF
    document = OCRDocument(
        file_path="complex_scanned_document.pdf",
        enhance_quality=True,
        preserve_layout=True
    )
    
    # Index the document
    result = await orchestrator.process_and_index(document)
    print(f"Processed {result.pages_processed} pages")
    print(f"Average OCR confidence: {result.avg_confidence:.2f}")
    
    # Query the processed content
    response = await orchestrator.query_and_generate(
        query="What are the key findings in this document?",
        min_confidence=0.8,  # Only use high-confidence OCR results
        preserve_layout=True
    )
    
    print(response.generated_answer)

asyncio.run(main())
```

## 📋 Detailed Features

### OCR Engine Comparison

| Engine | Speed | Accuracy | Language Support | Layout Awareness |
|--------|-------|----------|------------------|------------------|
| Tesseract | ⭐⭐⭐ | ⭐⭐⭐⭐ | 100+ languages | ⭐⭐ |
| EasyOCR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 80+ languages | ⭐⭐⭐ |
| AWS Textract | ⭐⭐ | ⭐⭐⭐⭐⭐ | English + others | ⭐⭐⭐⭐⭐ |

### Preprocessing Pipeline

1. **Image Enhancement**
   - Noise reduction using adaptive filters
   - Contrast adjustment and gamma correction
   - Binarization with adaptive thresholding

2. **Geometric Correction**
   - Automatic rotation detection and correction
   - Deskewing using Hough transform
   - Perspective correction for mobile photos

3. **Resolution Optimization**
   - Multi-scale processing
   - Super-resolution enhancement
   - Optimal DPI detection

### Quality Assessment

```python
# Assess OCR quality
quality_report = await orchestrator.assess_ocr_quality(document)
print(f"Overall confidence: {quality_report.overall_confidence}")
print(f"Text coverage: {quality_report.text_coverage_ratio}")
print(f"Layout preservation: {quality_report.layout_score}")
```

## 🔧 Advanced Configuration

### Custom OCR Pipeline

```python
from src.ocr_processors import create_ocr_pipeline

# Create custom OCR pipeline
ocr_pipeline = create_ocr_pipeline([
    {"engine": "easyocr", "confidence_threshold": 0.8},
    {"engine": "tesseract", "fallback": True},
    {"engine": "aws_textract", "for_tables": True}
])
```

### Layout-Aware Chunking

```python
from src.chunkers import LayoutAwareChunker

chunker = LayoutAwareChunker(
    preserve_headers=True,
    preserve_tables=True,
    min_chunk_size=200,
    max_chunk_size=1000,
    confidence_threshold=0.7
)
```

## 📊 Use Cases

### 1. Legal Document Processing
- Contract analysis with clause detection
- Legal precedent research
- Compliance document processing

### 2. Medical Records
- Clinical note extraction
- Lab report processing
- Medical form digitization

### 3. Financial Documents
- Invoice and receipt processing
- Bank statement analysis
- Financial report extraction

### 4. Research Papers
- Academic paper digitization
- Citation extraction
- Figure and table processing

## 🧪 Examples

See the `examples/` directory for comprehensive demonstrations:

- `basic_pdf_processing.py`: Simple PDF OCR and RAG
- `batch_document_processing.py`: Large-scale document processing
- `quality_assessment_demo.py`: OCR quality evaluation
- `layout_preservation_example.py`: Structure-aware processing

## 🔍 Monitoring and Debugging

### OCR Quality Metrics

```python
# Monitor processing quality
metrics = await orchestrator.get_processing_metrics()
print(f"Success rate: {metrics.success_rate:.2f}")
print(f"Average processing time: {metrics.avg_processing_time:.2f}s")
print(f"Confidence distribution: {metrics.confidence_histogram}")
```

### Error Handling

The system includes comprehensive error handling for:
- Corrupted or unreadable files
- OCR engine failures
- Layout detection errors
- Vector store connection issues

## 🛠️ Development

### Adding Custom OCR Engines

```python
from src.ocr_processors.base import BaseOCRProcessor

class CustomOCRProcessor(BaseOCRProcessor):
    async def extract_text(self, image_data: bytes) -> OCRResult:
        # Implement your OCR logic
        pass
```

### Custom Chunking Strategies

```python
from src.chunkers.base import BaseOCRChunker

class CustomChunker(BaseOCRChunker):
    def chunk_with_layout(self, ocr_result: OCRResult) -> List[OCRChunk]:
        # Implement your chunking logic
        pass
```

## 📈 Performance Optimization

### Batch Processing

```python
# Process multiple documents efficiently
documents = [OCRDocument(path=p) for p in pdf_paths]
results = await orchestrator.batch_process(
    documents,
    batch_size=10,
    max_concurrent=3
)
```

### Caching

```python
# Enable OCR result caching
orchestrator.configure(
    enable_ocr_cache=True,
    cache_duration_hours=24,
    cache_backend="redis"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all quality checks pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

### Common Issues

**Q: OCR results are poor quality?**
A: Try enabling image preprocessing and using multiple OCR engines with fallback.

**Q: Layout is not preserved?**
A: Use the layout-aware chunker and enable structure preservation options.

**Q: Processing is slow?**
A: Enable batch processing and adjust concurrent processing limits.

### Performance Tips

- Use appropriate image resolution (300 DPI for text)
- Enable preprocessing for scanned documents
- Use confidence thresholds to filter low-quality results
- Consider cloud OCR services for high-accuracy requirements

## 🔮 Roadmap

- [ ] Support for handwritten text recognition
- [ ] Advanced table extraction with relationships
- [ ] Multi-modal embeddings for images and text
- [ ] Real-time OCR processing pipeline
- [ ] Integration with more cloud OCR services
