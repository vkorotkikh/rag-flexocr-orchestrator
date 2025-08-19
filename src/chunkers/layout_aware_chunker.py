"""
Layout-aware chunker that preserves document structure and hierarchy.
"""

from typing import List, Dict, Any, Optional
import logging

from .base import BaseOCRChunker
from ..models import OCRResult, OCRChunk, ChunkingStrategy, OCRRegion, RegionType

logger = logging.getLogger(__name__)


class LayoutAwareChunker(BaseOCRChunker):
    """
    Chunks documents while preserving layout structure and hierarchy.
    
    This chunker respects document structure elements like headers, paragraphs,
    lists, and tables, ensuring that semantically related content stays together.
    """
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 confidence_threshold: float = 0.6,
                 preserve_headers: bool = True,
                 preserve_tables: bool = True,
                 preserve_lists: bool = True,
                 max_header_chunk_size: int = 500,
                 respect_column_boundaries: bool = True):
        """
        Initialize layout-aware chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size in characters
            confidence_threshold: Minimum OCR confidence for inclusion
            preserve_headers: Keep headers with following content
            preserve_tables: Keep table content together
            preserve_lists: Keep list items together
            max_header_chunk_size: Maximum size for header-only chunks
            respect_column_boundaries: Respect column layout boundaries
        """
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            confidence_threshold=confidence_threshold,
            preserve_structure=True
        )
        
        self.preserve_headers = preserve_headers
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        self.max_header_chunk_size = max_header_chunk_size
        self.respect_column_boundaries = respect_column_boundaries
    
    def get_strategy_name(self) -> ChunkingStrategy:
        """Get the chunking strategy name."""
        return ChunkingStrategy.LAYOUT_AWARE
    
    def chunk_ocr_result(self, ocr_result: OCRResult) -> List[OCRChunk]:
        """
        Chunk OCR result while preserving layout structure.
        
        Args:
            ocr_result: OCR result to chunk
            
        Returns:
            List of layout-aware OCR chunks
        """
        self.logger.info(f"Starting layout-aware chunking for page {ocr_result.page_number}")
        
        # Filter regions by confidence
        filtered_regions = self._filter_regions_by_confidence(ocr_result.regions)
        
        if not filtered_regions:
            self.logger.warning("No regions passed confidence filter")
            return []
        
        # Organize regions by structure
        structured_regions = self._organize_regions_by_structure(filtered_regions)
        
        # Handle tables separately if they exist
        table_chunks = []
        if self.preserve_tables and ocr_result.tables:
            table_chunks = self._chunk_tables(ocr_result.tables, ocr_result)
        
        # Chunk text regions while preserving layout
        text_chunks = self._chunk_structured_regions(structured_regions, ocr_result)
        
        # Combine and sort all chunks by reading order
        all_chunks = text_chunks + table_chunks
        all_chunks.sort(key=lambda c: c.metadata.reading_order or float('inf'))
        
        # Update chunk indices
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = idx
        
        self.logger.info(f"Created {len(all_chunks)} layout-aware chunks")
        return all_chunks
    
    def _organize_regions_by_structure(self, regions: List[OCRRegion]) -> Dict[str, List[OCRRegion]]:
        """Organize regions by their structural role."""
        structured = {
            'headers': [],
            'titles': [],
            'paragraphs': [],
            'lists': [],
            'captions': [],
            'footers': [],
            'other': []
        }
        
        for region in regions:
            region_type = region.region_type.value
            
            if region_type == 'header':
                structured['headers'].append(region)
            elif region_type == 'title':
                structured['titles'].append(region)
            elif region_type == 'paragraph':
                structured['paragraphs'].append(region)
            elif region_type == 'list':
                structured['lists'].append(region)
            elif region_type == 'caption':
                structured['captions'].append(region)
            elif region_type == 'footer':
                structured['footers'].append(region)
            else:
                structured['other'].append(region)
        
        # Sort each category by reading order
        for category in structured.values():
            category.sort(key=lambda r: (r.reading_order or float('inf'), r.bounding_box.y1, r.bounding_box.x1))
        
        return structured
    
    def _chunk_structured_regions(self, structured_regions: Dict[str, List[OCRRegion]], ocr_result: OCRResult) -> List[OCRChunk]:
        """Chunk regions while preserving structure."""
        chunks = []
        
        # Process in hierarchical order
        processing_order = ['headers', 'titles', 'paragraphs', 'lists', 'captions', 'other', 'footers']
        
        current_context = {
            'current_header': None,
            'current_section_content': [],
            'chunk_index': 0
        }
        
        for category in processing_order:
            regions = structured_regions.get(category, [])
            
            if category == 'headers':
                chunks.extend(self._process_headers(regions, current_context, ocr_result))
            elif category == 'titles':
                chunks.extend(self._process_titles(regions, current_context, ocr_result))
            elif category == 'paragraphs':
                chunks.extend(self._process_paragraphs(regions, current_context, ocr_result))
            elif category == 'lists':
                chunks.extend(self._process_lists(regions, current_context, ocr_result))
            elif category == 'captions':
                chunks.extend(self._process_captions(regions, current_context, ocr_result))
            elif category == 'footers':
                chunks.extend(self._process_footers(regions, current_context, ocr_result))
            else:
                chunks.extend(self._process_other_regions(regions, current_context, ocr_result))
        
        return chunks
    
    def _process_headers(self, headers: List[OCRRegion], context: Dict, ocr_result: OCRResult) -> List[OCRChunk]:
        """Process header regions."""
        chunks = []
        
        for header in headers:
            # Headers can be standalone or combined with following content
            header_content = header.text.strip()
            
            if self.preserve_headers and len(header_content) < self.max_header_chunk_size:
                # Try to include following content with header
                context['current_header'] = header
                
                # Create header chunk if it's substantial enough
                if len(header_content) >= self.min_chunk_size:
                    chunk = self._create_ocr_chunk(
                        content=header_content,
                        regions=[header],
                        document_id=ocr_result.document_id,
                        page_number=ocr_result.page_number,
                        chunk_index=context['chunk_index'],
                        processing_notes=[f"Header chunk: {header.region_type.value}"]
                    )
                    chunks.append(chunk)
                    context['chunk_index'] += 1
            else:
                # Large header - treat as regular content
                chunk = self._create_ocr_chunk(
                    content=header_content,
                    regions=[header],
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Large header processed as content"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
        
        return chunks
    
    def _process_titles(self, titles: List[OCRRegion], context: Dict, ocr_result: OCRResult) -> List[OCRChunk]:
        """Process title regions."""
        chunks = []
        
        for title in titles:
            title_content = title.text.strip()
            
            # Titles are usually short and standalone
            if len(title_content) >= self.min_chunk_size:
                chunk = self._create_ocr_chunk(
                    content=title_content,
                    regions=[title],
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Title chunk"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
        
        return chunks
    
    def _process_paragraphs(self, paragraphs: List[OCRRegion], context: Dict, ocr_result: OCRResult) -> List[OCRChunk]:
        """Process paragraph regions with intelligent grouping."""
        chunks = []
        
        if not paragraphs:
            return chunks
        
        # Group paragraphs that belong together
        paragraph_groups = self._group_related_paragraphs(paragraphs)
        
        for group in paragraph_groups:
            # Include current header if available
            regions_to_include = []
            if context.get('current_header') and self.preserve_headers:
                regions_to_include.append(context['current_header'])
                context['current_header'] = None  # Use header only once
            
            regions_to_include.extend(group)
            
            # Chunk the group respecting size limits
            group_chunks = self._chunk_region_group(regions_to_include, ocr_result, context)
            chunks.extend(group_chunks)
        
        return chunks
    
    def _process_lists(self, lists: List[OCRRegion], context: Dict, ocr_result: OCRResult) -> List[OCRChunk]:
        """Process list regions keeping related items together."""
        chunks = []
        
        if not lists:
            return chunks
        
        if self.preserve_lists:
            # Group consecutive list items
            list_groups = self._group_consecutive_list_items(lists)
            
            for group in list_groups:
                # Keep list items together if possible
                total_content = ' '.join(region.text for region in group)
                
                if len(total_content) <= self.chunk_size:
                    # Entire list fits in one chunk
                    chunk = self._create_ocr_chunk(
                        content=total_content,
                        regions=group,
                        document_id=ocr_result.document_id,
                        page_number=ocr_result.page_number,
                        chunk_index=context['chunk_index'],
                        processing_notes=["List items grouped together"]
                    )
                    chunks.append(chunk)
                    context['chunk_index'] += 1
                else:
                    # Split large lists while preserving item boundaries
                    sub_chunks = self._chunk_large_list(group, ocr_result, context)
                    chunks.extend(sub_chunks)
        else:
            # Process list items individually
            for list_item in lists:
                if len(list_item.text.strip()) >= self.min_chunk_size:
                    chunk = self._create_ocr_chunk(
                        content=list_item.text.strip(),
                        regions=[list_item],
                        document_id=ocr_result.document_id,
                        page_number=ocr_result.page_number,
                        chunk_index=context['chunk_index'],
                        processing_notes=["Individual list item"]
                    )
                    chunks.append(chunk)
                    context['chunk_index'] += 1
        
        return chunks
    
    def _process_captions(self, captions: List[OCRRegion], context: Dict, ocr_result: OCRResult) -> List[OCRChunk]:
        """Process caption regions."""
        chunks = []
        
        for caption in captions:
            caption_content = caption.text.strip()
            
            if len(caption_content) >= self.min_chunk_size:
                chunk = self._create_ocr_chunk(
                    content=caption_content,
                    regions=[caption],
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Caption chunk"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
        
        return chunks
    
    def _process_footers(self, footers: List[OCRRegion], context: Dict, ocr_result: OCRResult) -> List[OCRChunk]:
        """Process footer regions."""
        chunks = []
        
        # Combine all footers for a page
        if footers:
            footer_content = ' '.join(footer.text.strip() for footer in footers)
            
            if len(footer_content) >= self.min_chunk_size:
                chunk = self._create_ocr_chunk(
                    content=footer_content,
                    regions=footers,
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Combined footer content"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
        
        return chunks
    
    def _process_other_regions(self, other_regions: List[OCRRegion], context: Dict, ocr_result: OCRResult) -> List[OCRChunk]:
        """Process unclassified regions."""
        chunks = []
        
        # Group other regions by proximity
        region_groups = self._group_regions_by_proximity(other_regions)
        
        for group in region_groups:
            group_chunks = self._chunk_region_group(group, ocr_result, context)
            chunks.extend(group_chunks)
        
        return chunks
    
    def _group_related_paragraphs(self, paragraphs: List[OCRRegion]) -> List[List[OCRRegion]]:
        """Group paragraphs that should be chunked together."""
        if not paragraphs:
            return []
        
        groups = []
        current_group = [paragraphs[0]]
        
        for i in range(1, len(paragraphs)):
            prev_para = paragraphs[i-1]
            curr_para = paragraphs[i]
            
            # Check if paragraphs should be grouped
            if self._should_group_paragraphs(prev_para, curr_para):
                current_group.append(curr_para)
            else:
                groups.append(current_group)
                current_group = [curr_para]
        
        groups.append(current_group)
        return groups
    
    def _should_group_paragraphs(self, para1: OCRRegion, para2: OCRRegion) -> bool:
        """Determine if two paragraphs should be grouped together."""
        # Check vertical proximity
        vertical_gap = para2.bounding_box.y1 - para1.bounding_box.y2
        avg_height = (para1.bounding_box.height + para2.bounding_box.height) / 2
        
        # Group if gap is small relative to paragraph height
        if vertical_gap < avg_height * 0.5:
            return True
        
        # Check horizontal alignment (same column)
        horizontal_overlap = min(para1.bounding_box.x2, para2.bounding_box.x2) - max(para1.bounding_box.x1, para2.bounding_box.x1)
        min_width = min(para1.bounding_box.width, para2.bounding_box.width)
        
        # Group if they're in the same column
        if horizontal_overlap > min_width * 0.7:
            return True
        
        return False
    
    def _group_consecutive_list_items(self, list_items: List[OCRRegion]) -> List[List[OCRRegion]]:
        """Group consecutive list items together."""
        if not list_items:
            return []
        
        groups = []
        current_group = [list_items[0]]
        
        for i in range(1, len(list_items)):
            prev_item = list_items[i-1]
            curr_item = list_items[i]
            
            # Check if items are consecutive
            vertical_gap = curr_item.bounding_box.y1 - prev_item.bounding_box.y2
            avg_height = (prev_item.bounding_box.height + curr_item.bounding_box.height) / 2
            
            if vertical_gap < avg_height:  # Items are close vertically
                current_group.append(curr_item)
            else:
                groups.append(current_group)
                current_group = [curr_item]
        
        groups.append(current_group)
        return groups
    
    def _group_regions_by_proximity(self, regions: List[OCRRegion]) -> List[List[OCRRegion]]:
        """Group regions by spatial proximity."""
        if not regions:
            return []
        
        # Sort by position
        sorted_regions = sorted(regions, key=lambda r: (r.bounding_box.y1, r.bounding_box.x1))
        
        groups = []
        current_group = [sorted_regions[0]]
        
        for i in range(1, len(sorted_regions)):
            prev_region = sorted_regions[i-1]
            curr_region = sorted_regions[i]
            
            # Check proximity
            if self._are_regions_proximate(prev_region, curr_region):
                current_group.append(curr_region)
            else:
                groups.append(current_group)
                current_group = [curr_region]
        
        groups.append(current_group)
        return groups
    
    def _are_regions_proximate(self, region1: OCRRegion, region2: OCRRegion, proximity_threshold: float = 2.0) -> bool:
        """Check if two regions are spatially proximate."""
        bbox1, bbox2 = region1.bounding_box, region2.bounding_box
        
        # Calculate distance between regions
        center1 = bbox1.center
        center2 = bbox2.center
        
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        avg_dimension = (bbox1.width + bbox1.height + bbox2.width + bbox2.height) / 4
        
        return distance < avg_dimension * proximity_threshold
    
    def _chunk_region_group(self, regions: List[OCRRegion], ocr_result: OCRResult, context: Dict) -> List[OCRChunk]:
        """Chunk a group of regions respecting size limits."""
        chunks = []
        
        # Combine text from all regions
        combined_text = ' '.join(region.text.strip() for region in regions)
        
        if len(combined_text) <= self.chunk_size:
            # Entire group fits in one chunk
            chunk = self._create_ocr_chunk(
                content=combined_text,
                regions=regions,
                document_id=ocr_result.document_id,
                page_number=ocr_result.page_number,
                chunk_index=context['chunk_index'],
                processing_notes=["Grouped regions within size limit"]
            )
            chunks.append(chunk)
            context['chunk_index'] += 1
        else:
            # Need to split the group
            sub_chunks = self._split_region_group(regions, ocr_result, context)
            chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_region_group(self, regions: List[OCRRegion], ocr_result: OCRResult, context: Dict) -> List[OCRChunk]:
        """Split a large group of regions into multiple chunks."""
        chunks = []
        current_chunk_regions = []
        current_chunk_length = 0
        
        for region in regions:
            region_text = region.text.strip()
            region_length = len(region_text)
            
            # Check if adding this region would exceed chunk size
            if current_chunk_length + region_length > self.chunk_size and current_chunk_regions:
                # Create chunk with current regions
                chunk_content = ' '.join(r.text.strip() for r in current_chunk_regions)
                chunk = self._create_ocr_chunk(
                    content=chunk_content,
                    regions=current_chunk_regions,
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Split from large region group"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
                
                # Start new chunk with overlap if configured
                if self.chunk_overlap > 0 and current_chunk_regions:
                    # Include some regions for overlap
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
                chunk = self._create_ocr_chunk(
                    content=chunk_content,
                    regions=current_chunk_regions,
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Final chunk from split group"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
        
        return chunks
    
    def _get_overlap_regions(self, regions: List[OCRRegion], overlap_chars: int) -> List[OCRRegion]:
        """Get regions for overlap based on character count."""
        if not regions or overlap_chars <= 0:
            return []
        
        # Get regions from the end for overlap
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
    
    def _chunk_tables(self, tables: List, ocr_result: OCRResult) -> List[OCRChunk]:
        """Create chunks for table content."""
        chunks = []
        
        for table_idx, table in enumerate(tables):
            # Convert table to text representation
            table_text = self._table_to_text(table)
            
            if len(table_text.strip()) >= self.min_chunk_size:
                # Create pseudo-regions for table
                table_region = OCRRegion(
                    text=table_text,
                    confidence=0.9,  # Tables usually have good structure
                    bounding_box=self._get_table_bounding_box(table),
                    region_type=RegionType.TABLE,
                    region_id=f"table_{table_idx}"
                )
                
                chunk = self._create_ocr_chunk(
                    content=table_text,
                    regions=[table_region],
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=0,  # Will be updated later
                    processing_notes=[f"Table {table_idx} content"]
                )
                chunks.append(chunk)
        
        return chunks
    
    def _table_to_text(self, table) -> str:
        """Convert table structure to readable text."""
        try:
            # Simple table to text conversion
            text_lines = []
            
            if hasattr(table, 'to_text_matrix'):
                matrix = table.to_text_matrix()
                for row in matrix:
                    row_text = ' | '.join(str(cell) for cell in row)
                    text_lines.append(row_text)
            else:
                # Fallback for different table structures
                text_lines.append("Table content detected")
            
            return '\n'.join(text_lines)
        except Exception as e:
            self.logger.warning(f"Failed to convert table to text: {e}")
            return "Table content (conversion failed)"
    
    def _get_table_bounding_box(self, table):
        """Get bounding box for table."""
        try:
            if hasattr(table, 'cells') and table.cells:
                # Calculate overall bounding box from cells
                min_x1 = min(cell.bounding_box.x1 for cell in table.cells)
                min_y1 = min(cell.bounding_box.y1 for cell in table.cells)
                max_x2 = max(cell.bounding_box.x2 for cell in table.cells)
                max_y2 = max(cell.bounding_box.y2 for cell in table.cells)
                
                from ..models import BoundingBox
                return BoundingBox(x1=min_x1, y1=min_y1, x2=max_x2, y2=max_y2)
        except:
            pass
        
        # Fallback bounding box
        from ..models import BoundingBox
        return BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=100.0)
    
    def _chunk_large_list(self, list_items: List[OCRRegion], ocr_result: OCRResult, context: Dict) -> List[OCRChunk]:
        """Chunk a large list while preserving item boundaries."""
        chunks = []
        current_items = []
        current_length = 0
        
        for item in list_items:
            item_text = item.text.strip()
            item_length = len(item_text)
            
            if current_length + item_length > self.chunk_size and current_items:
                # Create chunk with current items
                chunk_content = ' '.join(item.text.strip() for item in current_items)
                chunk = self._create_ocr_chunk(
                    content=chunk_content,
                    regions=current_items,
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Partial list content"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
                
                current_items = [item]
                current_length = item_length
            else:
                current_items.append(item)
                current_length += item_length
        
        # Create final chunk
        if current_items:
            chunk_content = ' '.join(item.text.strip() for item in current_items)
            if len(chunk_content.strip()) >= self.min_chunk_size:
                chunk = self._create_ocr_chunk(
                    content=chunk_content,
                    regions=current_items,
                    document_id=ocr_result.document_id,
                    page_number=ocr_result.page_number,
                    chunk_index=context['chunk_index'],
                    processing_notes=["Final list content"]
                )
                chunks.append(chunk)
                context['chunk_index'] += 1
        
        return chunks
