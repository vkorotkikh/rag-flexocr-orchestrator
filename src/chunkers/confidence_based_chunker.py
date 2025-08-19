"""
Confidence-based chunker that groups content by OCR reliability levels.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

from .base import BaseOCRChunker
from ..models import OCRResult, OCRChunk, ChunkingStrategy, OCRRegion

logger = logging.getLogger(__name__)


class ConfidenceBasedChunker(BaseOCRChunker):
    """
    Chunks documents based on OCR confidence levels.
    
    This chunker groups text regions with similar confidence scores together,
    ensuring that high-quality content is separated from lower-quality content
    for more reliable RAG retrieval.
    """
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 confidence_threshold: float = 0.6,
                 high_confidence_threshold: float = 0.8,
                 medium_confidence_threshold: float = 0.6,
                 low_confidence_threshold: float = 0.4,
                 separate_confidence_levels: bool = True,
                 confidence_variance_threshold: float = 0.2):
        """
        Initialize confidence-based chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size in characters
            confidence_threshold: Minimum OCR confidence for inclusion
            high_confidence_threshold: Threshold for high-confidence content
            medium_confidence_threshold: Threshold for medium-confidence content
            low_confidence_threshold: Threshold for low-confidence content
            separate_confidence_levels: Keep different confidence levels in separate chunks
            confidence_variance_threshold: Maximum allowed confidence variance within a chunk
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            confidence_threshold=confidence_threshold,
            preserve_structure=False  # Structure is secondary to confidence
        )
        
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.separate_confidence_levels = separate_confidence_levels
        self.confidence_variance_threshold = confidence_variance_threshold
    
    def get_strategy_name(self) -> ChunkingStrategy:
        """Get the chunking strategy name."""
        return ChunkingStrategy.CONFIDENCE_BASED
    
    def chunk_ocr_result(self, ocr_result: OCRResult) -> List[OCRChunk]:
        """
        Chunk OCR result based on confidence levels.
        
        Args:
            ocr_result: OCR result to chunk
            
        Returns:
            List of confidence-based OCR chunks
        """
        self.logger.info(f"Starting confidence-based chunking for page {ocr_result.page_number}")
        
        # Filter regions by minimum confidence
        filtered_regions = self._filter_regions_by_confidence(ocr_result.regions)
        
        if not filtered_regions:
            self.logger.warning("No regions passed confidence filter")
            return []
        
        # Categorize regions by confidence level
        confidence_categories = self._categorize_regions_by_confidence(filtered_regions)
        
        # Chunk each confidence category separately if configured
        all_chunks = []
        
        if self.separate_confidence_levels:
            for category, regions in confidence_categories.items():
                if regions:
                    category_chunks = self._chunk_confidence_category(
                        regions, category, ocr_result
                    )
                    all_chunks.extend(category_chunks)
        else:
            # Chunk all regions together but track confidence variance
            all_chunks = self._chunk_mixed_confidence_regions(filtered_regions, ocr_result)
        
        # Sort chunks by reading order and position
        all_chunks.sort(key=lambda c: (
            c.metadata.reading_order or float('inf'),
            c.metadata.bounding_boxes[0].y1 if c.metadata.bounding_boxes else 0,
            c.metadata.bounding_boxes[0].x1 if c.metadata.bounding_boxes else 0
        ))
        
        # Update chunk indices
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = idx
        
        self.logger.info(f"Created {len(all_chunks)} confidence-based chunks")
        return all_chunks
    
    def _categorize_regions_by_confidence(self, regions: List[OCRRegion]) -> Dict[str, List[OCRRegion]]:
        """Categorize regions by confidence levels."""
        categories = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for region in regions:
            if region.confidence >= self.high_confidence_threshold:
                categories['high'].append(region)
            elif region.confidence >= self.medium_confidence_threshold:
                categories['medium'].append(region)
            elif region.confidence >= self.low_confidence_threshold:
                categories['low'].append(region)
            # Regions below low threshold are already filtered out
        
        # Sort each category by reading order and position
        for category_regions in categories.values():
            category_regions.sort(key=lambda r: (
                r.reading_order or float('inf'),
                r.bounding_box.y1,
                r.bounding_box.x1
            ))
        
        self.logger.debug(f"Confidence distribution - High: {len(categories['high'])}, "
                         f"Medium: {len(categories['medium'])}, Low: {len(categories['low'])}")
        
        return categories
    
    def _chunk_confidence_category(self, regions: List[OCRRegion], category: str, ocr_result: OCRResult) -> List[OCRChunk]:
        """Chunk regions within a specific confidence category."""
        chunks = []
        
        if not regions:
            return chunks
        
        self.logger.debug(f"Chunking {len(regions)} regions in {category} confidence category")
        
        # Group regions by spatial proximity within the confidence category
        region_groups = self._group_regions_by_proximity_and_confidence(regions)
        
        chunk_index = 0
        for group in region_groups:
            group_chunks = self._chunk_confidence_group(group, category, ocr_result, chunk_index)
            chunks.extend(group_chunks)
            chunk_index += len(group_chunks)
        
        return chunks
    
    def _chunk_mixed_confidence_regions(self, regions: List[OCRRegion], ocr_result: OCRResult) -> List[OCRChunk]:
        """Chunk regions with mixed confidence levels while monitoring variance."""
        chunks = []
        current_group = []
        current_confidences = []
        
        # Sort regions by reading order and position
        sorted_regions = sorted(regions, key=lambda r: (
            r.reading_order or float('inf'),
            r.bounding_box.y1,
            r.bounding_box.x1
        ))
        
        chunk_index = 0
        
        for region in sorted_regions:
            # Check if adding this region would exceed confidence variance threshold
            test_confidences = current_confidences + [region.confidence]
            confidence_variance = self._calculate_variance(test_confidences)
            
            current_text_length = sum(len(r.text) for r in current_group)
            
            # Decide whether to add region to current group or start new chunk
            should_start_new_chunk = (
                # Variance too high
                confidence_variance > self.confidence_variance_threshold or
                # Size limit exceeded
                (current_text_length + len(region.text) > self.chunk_size and current_group) or
                # Large confidence gap
                (current_confidences and abs(region.confidence - sum(current_confidences)/len(current_confidences)) > 0.3)
            )
            
            if should_start_new_chunk and current_group:
                # Create chunk with current group
                chunk = self._create_confidence_chunk(
                    current_group, "mixed", ocr_result, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new group with overlap if configured
                if self.chunk_overlap > 0:
                    overlap_regions, overlap_confidences = self._get_overlap_regions_with_confidence(
                        current_group, current_confidences, self.chunk_overlap
                    )
                    current_group = overlap_regions + [region]
                    current_confidences = overlap_confidences + [region.confidence]
                else:
                    current_group = [region]
                    current_confidences = [region.confidence]
            else:
                current_group.append(region)
                current_confidences.append(region.confidence)
        
        # Create final chunk if there are remaining regions
        if current_group:
            chunk = self._create_confidence_chunk(
                current_group, "mixed", ocr_result, chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _group_regions_by_proximity_and_confidence(self, regions: List[OCRRegion]) -> List[List[OCRRegion]]:
        """Group regions by both spatial proximity and confidence similarity."""
        if not regions:
            return []
        
        groups = []
        current_group = [regions[0]]
        
        for i in range(1, len(regions)):
            prev_region = regions[i-1]
            curr_region = regions[i]
            
            # Check if regions should be grouped
            should_group = (
                self._are_regions_spatially_close(prev_region, curr_region) and
                self._are_confidence_levels_similar(prev_region.confidence, curr_region.confidence)
            )
            
            if should_group:
                current_group.append(curr_region)
            else:
                groups.append(current_group)
                current_group = [curr_region]
        
        groups.append(current_group)
        return groups
    
    def _are_regions_spatially_close(self, region1: OCRRegion, region2: OCRRegion, proximity_factor: float = 2.0) -> bool:
        """Check if two regions are spatially close."""
        bbox1, bbox2 = region1.bounding_box, region2.bounding_box
        
        # Calculate vertical and horizontal distances
        vertical_distance = abs(bbox1.y1 - bbox2.y1)
        horizontal_distance = abs(bbox1.x1 - bbox2.x1)
        
        # Average dimensions for reference
        avg_height = (bbox1.height + bbox2.height) / 2
        avg_width = (bbox1.width + bbox2.width) / 2
        
        # Regions are close if distances are within reasonable multiples of their dimensions
        return (vertical_distance < avg_height * proximity_factor and
                horizontal_distance < avg_width * proximity_factor)
    
    def _are_confidence_levels_similar(self, conf1: float, conf2: float, threshold: float = 0.2) -> bool:
        """Check if two confidence levels are similar."""
        return abs(conf1 - conf2) <= threshold
    
    def _chunk_confidence_group(self, regions: List[OCRRegion], category: str, ocr_result: OCRResult, start_index: int) -> List[OCRChunk]:
        """Chunk a group of regions with similar confidence."""
        chunks = []
        
        # Calculate total content length
        total_content = ' '.join(region.text.strip() for region in regions)
        
        if len(total_content) <= self.chunk_size:
            # Entire group fits in one chunk
            chunk = self._create_confidence_chunk(regions, category, ocr_result, start_index)
            chunks.append(chunk)
        else:
            # Split group into multiple chunks
            sub_chunks = self._split_confidence_group(regions, category, ocr_result, start_index)
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_confidence_group(self, regions: List[OCRRegion], category: str, ocr_result: OCRResult, start_index: int) -> List[OCRChunk]:
        """Split a large confidence group into multiple chunks."""
        chunks = []
        current_chunk_regions = []
        current_chunk_length = 0
        chunk_index = start_index
        
        for region in regions:
            region_text = region.text.strip()
            region_length = len(region_text)
            
            # Check if adding this region would exceed chunk size
            if current_chunk_length + region_length > self.chunk_size and current_chunk_regions:
                # Create chunk with current regions
                chunk = self._create_confidence_chunk(
                    current_chunk_regions, category, ocr_result, chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap if configured
                if self.chunk_overlap > 0:
                    overlap_regions = self._get_overlap_regions(current_chunk_regions, self.chunk_overlap)
                    current_chunk_regions = overlap_regions + [region]
                    current_chunk_length = sum(len(r.text) for r in current_chunk_regions)
                else:
                    current_chunk_regions = [region]
                    current_chunk_length = region_length
            else:
                current_chunk_regions.append(region)
                current_chunk_length += region_length
        
        # Create final chunk if there are remaining regions
        if current_chunk_regions:
            chunk_content = ' '.join(r.text.strip() for r in current_chunk_regions)
            if len(chunk_content.strip()) >= self.min_chunk_size:
                chunk = self._create_confidence_chunk(
                    current_chunk_regions, category, ocr_result, chunk_index
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_confidence_chunk(self, regions: List[OCRRegion], category: str, ocr_result: OCRResult, chunk_index: int) -> OCRChunk:
        """Create a chunk with confidence-specific metadata."""
        content = ' '.join(region.text.strip() for region in regions)
        
        # Calculate confidence statistics
        confidences = [r.confidence for r in regions]
        confidence_stats = {
            'min': min(confidences),
            'max': max(confidences),
            'mean': sum(confidences) / len(confidences),
            'variance': self._calculate_variance(confidences)
        }
        
        processing_notes = [
            f"Confidence category: {category}",
            f"Confidence range: {confidence_stats['min']:.2f} - {confidence_stats['max']:.2f}",
            f"Average confidence: {confidence_stats['mean']:.2f}",
            f"Confidence variance: {confidence_stats['variance']:.3f}"
        ]
        
        chunk = self._create_ocr_chunk(
            content=content,
            regions=regions,
            document_id=ocr_result.document_id,
            page_number=ocr_result.page_number,
            chunk_index=chunk_index,
            processing_notes=processing_notes
        )
        
        return chunk
    
    def _get_overlap_regions(self, regions: List[OCRRegion], overlap_chars: int) -> List[OCRRegion]:
        """Get regions for overlap based on character count."""
        if not regions or overlap_chars <= 0:
            return []
        
        overlap_regions = []
        char_count = 0
        
        for region in reversed(regions):
            region_chars = len(region.text)
            if char_count + region_chars <= overlap_chars:
                overlap_regions.insert(0, region)
                char_count += region_chars
            else:
                break
        
        return overlap_regions
    
    def _get_overlap_regions_with_confidence(self, regions: List[OCRRegion], confidences: List[float], overlap_chars: int) -> Tuple[List[OCRRegion], List[float]]:
        """Get regions and their confidences for overlap."""
        if not regions or overlap_chars <= 0:
            return [], []
        
        overlap_regions = []
        overlap_confidences = []
        char_count = 0
        
        for region, conf in zip(reversed(regions), reversed(confidences)):
            region_chars = len(region.text)
            if char_count + region_chars <= overlap_chars:
                overlap_regions.insert(0, region)
                overlap_confidences.insert(0, conf)
                char_count += region_chars
            else:
                break
        
        return overlap_regions, overlap_confidences
    
    def get_confidence_distribution(self, chunks: List[OCRChunk]) -> Dict[str, Any]:
        """Get confidence distribution statistics for chunks."""
        if not chunks:
            return {}
        
        all_confidences = [chunk.metadata.average_confidence for chunk in chunks]
        
        high_conf_chunks = sum(1 for conf in all_confidences if conf >= self.high_confidence_threshold)
        medium_conf_chunks = sum(1 for conf in all_confidences if self.medium_confidence_threshold <= conf < self.high_confidence_threshold)
        low_conf_chunks = sum(1 for conf in all_confidences if self.low_confidence_threshold <= conf < self.medium_confidence_threshold)
        
        return {
            'total_chunks': len(chunks),
            'high_confidence_chunks': high_conf_chunks,
            'medium_confidence_chunks': medium_conf_chunks,
            'low_confidence_chunks': low_conf_chunks,
            'average_confidence': sum(all_confidences) / len(all_confidences),
            'min_confidence': min(all_confidences),
            'max_confidence': max(all_confidences),
            'confidence_variance': self._calculate_variance(all_confidences),
            'high_confidence_ratio': high_conf_chunks / len(chunks),
            'reliable_chunks_ratio': (high_conf_chunks + medium_conf_chunks) / len(chunks)
        }
