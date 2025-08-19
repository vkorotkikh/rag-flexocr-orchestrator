"""
Base chunker interface for OCR-aware document chunking.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import uuid
import logging
from datetime import datetime

from ..models import OCRResult, OCRChunk, ChunkMetadata, ChunkingStrategy, OCRRegion, BoundingBox

logger = logging.getLogger(__name__)


class BaseOCRChunker(ABC):
    """Base class for OCR-aware chunking strategies."""
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 confidence_threshold: float = 0.6,
                 preserve_structure: bool = True):
        """
        Initialize base OCR chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size in characters
            confidence_threshold: Minimum OCR confidence for inclusion
            preserve_structure: Whether to preserve document structure
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.confidence_threshold = confidence_threshold
        self.preserve_structure = preserve_structure
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def chunk_ocr_result(self, ocr_result: OCRResult) -> List[OCRChunk]:
        """
        Chunk an OCR result into smaller segments.
        
        Args:
            ocr_result: OCR result to chunk
            
        Returns:
            List of OCR chunks
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> ChunkingStrategy:
        """Get the chunking strategy name."""
        pass
    
    def chunk_multiple_results(self, ocr_results: List[OCRResult]) -> List[OCRChunk]:
        """
        Chunk multiple OCR results (e.g., multi-page documents).
        
        Args:
            ocr_results: List of OCR results to chunk
            
        Returns:
            List of OCR chunks from all pages
        """
        all_chunks = []
        
        for result in ocr_results:
            try:
                chunks = self.chunk_ocr_result(result)
                all_chunks.extend(chunks)
                self.logger.debug(f"Chunked page {result.page_number} into {len(chunks)} chunks")
            except Exception as e:
                self.logger.error(f"Failed to chunk page {result.page_number}: {e}")
        
        # Update chunk indices to be global across all pages
        for global_idx, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = global_idx
        
        self.logger.info(f"Created {len(all_chunks)} total chunks from {len(ocr_results)} pages")
        return all_chunks
    
    def _filter_regions_by_confidence(self, regions: List[OCRRegion]) -> List[OCRRegion]:
        """Filter regions by confidence threshold."""
        filtered = [r for r in regions if r.confidence >= self.confidence_threshold]
        self.logger.debug(f"Filtered {len(regions)} regions to {len(filtered)} by confidence >= {self.confidence_threshold}")
        return filtered
    
    def _create_chunk_metadata(self,
                              document_id: str,
                              page_number: int,
                              chunk_index: int,
                              regions: List[OCRRegion],
                              **kwargs) -> ChunkMetadata:
        """Create comprehensive chunk metadata."""
        
        # Calculate confidence metrics
        confidences = [r.confidence for r in regions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        min_confidence = min(confidences) if confidences else 0.0
        max_confidence = max(confidences) if confidences else 0.0
        confidence_variance = self._calculate_variance(confidences)
        
        # Extract bounding boxes and region types
        bounding_boxes = [r.bounding_box for r in regions]
        region_types = [r.region_type for r in regions]
        
        # Calculate content characteristics
        text_content = ' '.join(r.text for r in regions)
        word_count = len(text_content.split())
        character_count = len(text_content)
        
        # Analyze content structure
        has_tables = any(rt.value == 'table' for rt in region_types)
        has_figures = any(rt.value == 'figure' for rt in region_types)
        is_header = any(rt.value == 'header' for rt in region_types)
        is_footer = any(rt.value == 'footer' for rt in region_types)
        is_title = any(rt.value == 'title' for rt in region_types)
        
        # Detect language characteristics
        primary_language = self._detect_primary_language(text_content)
        contains_multiple_languages = self._detect_multiple_languages(regions)
        
        # Analyze formatting
        has_bold_text = any(getattr(r, 'is_bold', False) for r in regions)
        has_italic_text = any(getattr(r, 'is_italic', False) for r in regions)
        font_sizes = [getattr(r, 'font_size', None) for r in regions if getattr(r, 'font_size', None)]
        font_size_range = (min(font_sizes), max(font_sizes)) if font_sizes else None
        
        # Determine hierarchy level
        hierarchy_level = self._determine_hierarchy_level(regions, region_types)
        
        # Determine reading order
        reading_order = self._determine_reading_order(regions)
        
        return ChunkMetadata(
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            bounding_boxes=bounding_boxes,
            region_types=region_types,
            average_confidence=avg_confidence,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            confidence_variance=confidence_variance,
            word_count=word_count,
            character_count=character_count,
            has_tables=has_tables,
            has_figures=has_figures,
            is_header=is_header,
            is_footer=is_footer,
            is_title=is_title,
            reading_order=reading_order,
            hierarchy_level=hierarchy_level,
            primary_language=primary_language,
            contains_multiple_languages=contains_multiple_languages,
            font_size_range=font_size_range,
            has_bold_text=has_bold_text,
            has_italic_text=has_italic_text,
            chunking_strategy=self.get_strategy_name(),
            processing_notes=kwargs.get('processing_notes', [])
        )
    
    def _create_ocr_chunk(self,
                         content: str,
                         regions: List[OCRRegion],
                         document_id: str,
                         page_number: int,
                         chunk_index: int,
                         **kwargs) -> OCRChunk:
        """Create an OCR chunk with metadata."""
        
        chunk_id = str(uuid.uuid4())
        
        metadata = self._create_chunk_metadata(
            document_id=document_id,
            page_number=page_number,
            chunk_index=chunk_index,
            regions=regions,
            **kwargs
        )
        
        # Check if chunk should be filtered
        should_filter, filter_reason = self._should_filter_chunk(content, metadata)
        
        chunk = OCRChunk(
            content=content,
            chunk_id=chunk_id,
            metadata=metadata,
            is_filtered=should_filter,
            filter_reason=filter_reason,
            needs_review=self._needs_manual_review(content, metadata)
        )
        
        return chunk
    
    def _should_filter_chunk(self, content: str, metadata: ChunkMetadata) -> tuple[bool, Optional[str]]:
        """Determine if chunk should be filtered due to quality issues."""
        
        # Filter very short chunks
        if len(content.strip()) < self.min_chunk_size:
            return True, "Content too short"
        
        # Filter low confidence chunks
        if metadata.average_confidence < self.confidence_threshold:
            return True, f"Low confidence: {metadata.average_confidence:.2f}"
        
        # Filter chunks with too many single characters (likely OCR artifacts)
        words = content.split()
        single_char_words = sum(1 for word in words if len(word.strip()) == 1)
        if len(words) > 0 and single_char_words / len(words) > 0.5:
            return True, "Too many single-character artifacts"
        
        # Filter chunks that are mostly non-alphanumeric
        alphanumeric_chars = sum(1 for char in content if char.isalnum())
        if len(content) > 0 and alphanumeric_chars / len(content) < 0.3:
            return True, "Mostly non-alphanumeric content"
        
        return False, None
    
    def _needs_manual_review(self, content: str, metadata: ChunkMetadata) -> bool:
        """Determine if chunk needs manual review."""
        
        # High variance in confidence scores
        if metadata.confidence_variance > 0.3:
            return True
        
        # Mixed content types
        unique_types = set(metadata.region_types)
        if len(unique_types) > 3:
            return True
        
        # Very low minimum confidence
        if metadata.min_confidence < 0.3:
            return True
        
        return False
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _detect_primary_language(self, text: str) -> Optional[str]:
        """Detect primary language of text."""
        try:
            from langdetect import detect
            if text and len(text.strip()) > 20:
                return detect(text)
        except:
            pass
        return None
    
    def _detect_multiple_languages(self, regions: List[OCRRegion]) -> bool:
        """Detect if chunk contains multiple languages."""
        languages = set()
        
        for region in regions:
            if hasattr(region, 'language') and region.language:
                languages.add(region.language)
        
        return len(languages) > 1
    
    def _determine_hierarchy_level(self, regions: List[OCRRegion], region_types: List) -> Optional[int]:
        """Determine hierarchical level of chunk."""
        # Simple heuristic based on region types
        type_hierarchy = {
            'title': 0,
            'header': 1,
            'paragraph': 2,
            'list': 3,
            'caption': 4,
            'footer': 5
        }
        
        levels = [type_hierarchy.get(rt.value, 2) for rt in region_types]
        return min(levels) if levels else None
    
    def _determine_reading_order(self, regions: List[OCRRegion]) -> Optional[int]:
        """Determine reading order of chunk."""
        if regions:
            # Use the reading order of the first region
            first_region = min(regions, key=lambda r: r.reading_order or float('inf'))
            return first_region.reading_order
        return None
    
    def _combine_overlapping_regions(self, regions: List[OCRRegion]) -> List[OCRRegion]:
        """Combine regions that overlap significantly."""
        if not regions:
            return []
        
        # Sort by position
        sorted_regions = sorted(regions, key=lambda r: (r.bounding_box.y1, r.bounding_box.x1))
        combined = []
        current_group = [sorted_regions[0]]
        
        for region in sorted_regions[1:]:
            # Check if this region overlaps significantly with current group
            if self._has_significant_overlap(current_group[-1], region):
                current_group.append(region)
            else:
                # Combine current group and start new one
                if current_group:
                    combined_region = self._merge_regions(current_group)
                    combined.append(combined_region)
                current_group = [region]
        
        # Don't forget the last group
        if current_group:
            combined_region = self._merge_regions(current_group)
            combined.append(combined_region)
        
        return combined
    
    def _has_significant_overlap(self, region1: OCRRegion, region2: OCRRegion, threshold: float = 0.3) -> bool:
        """Check if two regions have significant overlap."""
        bbox1, bbox2 = region1.bounding_box, region2.bounding_box
        
        # Calculate intersection
        intersection = bbox1.intersection_with(bbox2)
        if not intersection:
            return False
        
        # Calculate overlap ratio
        intersection_area = intersection.area
        min_area = min(bbox1.area, bbox2.area)
        
        return intersection_area / min_area > threshold
    
    def _merge_regions(self, regions: List[OCRRegion]) -> OCRRegion:
        """Merge multiple regions into one."""
        if len(regions) == 1:
            return regions[0]
        
        # Combine text
        combined_text = ' '.join(r.text for r in regions)
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in regions) / len(regions)
        
        # Calculate combined bounding box
        min_x1 = min(r.bounding_box.x1 for r in regions)
        min_y1 = min(r.bounding_box.y1 for r in regions)
        max_x2 = max(r.bounding_box.x2 for r in regions)
        max_y2 = max(r.bounding_box.y2 for r in regions)
        
        combined_bbox = BoundingBox(x1=min_x1, y1=min_y1, x2=max_x2, y2=max_y2)
        
        # Use the most common region type
        region_types = [r.region_type for r in regions]
        most_common_type = max(set(region_types), key=region_types.count)
        
        # Use the earliest reading order
        reading_orders = [r.reading_order for r in regions if r.reading_order is not None]
        min_reading_order = min(reading_orders) if reading_orders else None
        
        return OCRRegion(
            text=combined_text,
            confidence=avg_confidence,
            bounding_box=combined_bbox,
            region_type=most_common_type,
            reading_order=min_reading_order,
            region_id=str(uuid.uuid4())
        )
    
    def get_chunking_stats(self, chunks: List[OCRChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        filtered_chunks = sum(1 for c in chunks if c.is_filtered)
        high_quality_chunks = sum(1 for c in chunks if c.is_high_confidence)
        needs_review_chunks = sum(1 for c in chunks if c.needs_review)
        
        confidences = [c.metadata.average_confidence for c in chunks]
        word_counts = [c.metadata.word_count for c in chunks]
        
        return {
            'total_chunks': total_chunks,
            'filtered_chunks': filtered_chunks,
            'high_quality_chunks': high_quality_chunks,
            'needs_review_chunks': needs_review_chunks,
            'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
            'min_confidence': min(confidences) if confidences else 0.0,
            'max_confidence': max(confidences) if confidences else 0.0,
            'average_word_count': sum(word_counts) / len(word_counts) if word_counts else 0.0,
            'total_words': sum(word_counts),
            'chunk_size_range': (min(word_counts), max(word_counts)) if word_counts else (0, 0)
        }
