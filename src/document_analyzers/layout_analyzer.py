"""
Layout analyzer for enhancing OCR results with structural information.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ..models import OCRResult, OCRRegion, RegionType, BoundingBox

logger = logging.getLogger(__name__)


class LayoutAnalyzer:
    """
    Analyzes document layout to enhance OCR results with structural information.
    
    Provides advanced layout detection including column analysis, reading order
    determination, and hierarchical structure identification.
    """
    
    def __init__(self,
                 enable_column_detection: bool = True,
                 enable_reading_order: bool = True,
                 enable_hierarchy_detection: bool = True,
                 min_column_width: float = 100.0,
                 column_gap_threshold: float = 30.0):
        """
        Initialize layout analyzer.
        
        Args:
            enable_column_detection: Enable multi-column layout detection
            enable_reading_order: Enable reading order determination
            enable_hierarchy_detection: Enable hierarchical structure detection
            min_column_width: Minimum width for column detection
            column_gap_threshold: Minimum gap between columns
        """
        self.enable_column_detection = enable_column_detection
        self.enable_reading_order = enable_reading_order
        self.enable_hierarchy_detection = enable_hierarchy_detection
        self.min_column_width = min_column_width
        self.column_gap_threshold = column_gap_threshold
        
        logger.info("Layout Analyzer initialized")
    
    async def analyze_layout(self, ocr_result: OCRResult) -> OCRResult:
        """
        Analyze and enhance OCR result with layout information.
        
        Args:
            ocr_result: Original OCR result
            
        Returns:
            Enhanced OCR result with layout analysis
        """
        logger.info(f"Analyzing layout for document {ocr_result.document_id}, page {ocr_result.page_number}")
        
        if not ocr_result.regions:
            logger.warning("No regions to analyze")
            return ocr_result
        
        # Create a copy to avoid modifying original
        enhanced_regions = [self._copy_region(region) for region in ocr_result.regions]
        
        # Detect columns if enabled
        if self.enable_column_detection:
            column_info = self._detect_columns(enhanced_regions)
            self._assign_column_info(enhanced_regions, column_info)
        
        # Determine reading order if enabled
        if self.enable_reading_order:
            self._determine_reading_order(enhanced_regions)
        
        # Detect hierarchical structure if enabled
        if self.enable_hierarchy_detection:
            self._detect_hierarchy(enhanced_regions)
        
        # Create enhanced OCR result
        enhanced_result = OCRResult(
            document_id=ocr_result.document_id,
            page_number=ocr_result.page_number,
            regions=enhanced_regions,
            tables=ocr_result.tables,
            ocr_engine=ocr_result.ocr_engine,
            processing_time=ocr_result.processing_time,
            image_dimensions=ocr_result.image_dimensions,
            language_detected=ocr_result.language_detected,
            overall_confidence=ocr_result.overall_confidence,
            text_coverage_ratio=ocr_result.text_coverage_ratio,
            low_confidence_regions=ocr_result.low_confidence_regions,
            preprocessing_applied=ocr_result.preprocessing_applied + ["layout_analysis"],
            confidence_threshold=ocr_result.confidence_threshold
        )
        
        logger.info("Layout analysis completed")
        return enhanced_result
    
    def _copy_region(self, region: OCRRegion) -> OCRRegion:
        """Create a copy of an OCR region."""
        return OCRRegion(
            text=region.text,
            confidence=region.confidence,
            bounding_box=BoundingBox(
                x1=region.bounding_box.x1,
                y1=region.bounding_box.y1,
                x2=region.bounding_box.x2,
                y2=region.bounding_box.y2
            ),
            region_type=region.region_type,
            language=region.language,
            font_size=region.font_size,
            is_bold=region.is_bold,
            is_italic=region.is_italic,
            reading_order=region.reading_order,
            parent_region_id=region.parent_region_id,
            child_region_ids=region.child_region_ids.copy(),
            region_id=region.region_id
        )
    
    def _detect_columns(self, regions: List[OCRRegion]) -> Dict[str, Any]:
        """Detect column layout in the document."""
        if not regions:
            return {'columns': 1, 'column_boundaries': []}
        
        # Extract x-coordinates of region boundaries
        left_edges = [region.bounding_box.x1 for region in regions]
        right_edges = [region.bounding_box.x2 for region in regions]
        
        # Find potential column boundaries by clustering x-coordinates
        column_boundaries = self._find_column_boundaries(left_edges, right_edges)
        
        # Validate columns based on width and content
        valid_columns = self._validate_columns(column_boundaries, regions)
        
        logger.debug(f"Detected {len(valid_columns)} columns")
        
        return {
            'columns': len(valid_columns),
            'column_boundaries': valid_columns,
            'column_gaps': self._calculate_column_gaps(valid_columns)
        }
    
    def _find_column_boundaries(self, left_edges: List[float], right_edges: List[float]) -> List[Tuple[float, float]]:
        """Find column boundaries using clustering."""
        if not left_edges:
            return []
        
        # Simple clustering approach
        sorted_lefts = sorted(set(left_edges))
        sorted_rights = sorted(set(right_edges))
        
        # Group nearby boundaries
        clustered_lefts = self._cluster_coordinates(sorted_lefts, self.column_gap_threshold)
        clustered_rights = self._cluster_coordinates(sorted_rights, self.column_gap_threshold)
        
        # Create column boundaries
        columns = []
        for left_cluster in clustered_lefts:
            left_boundary = min(left_cluster)
            
            # Find corresponding right boundary
            possible_rights = [r for r in clustered_rights if min(r) > left_boundary]
            if possible_rights:
                right_boundary = max(min(possible_rights))
                if right_boundary - left_boundary >= self.min_column_width:
                    columns.append((left_boundary, right_boundary))
        
        return columns
    
    def _cluster_coordinates(self, coordinates: List[float], threshold: float) -> List[List[float]]:
        """Cluster coordinates that are close together."""
        if not coordinates:
            return []
        
        clusters = []
        current_cluster = [coordinates[0]]
        
        for i in range(1, len(coordinates)):
            if coordinates[i] - coordinates[i-1] <= threshold:
                current_cluster.append(coordinates[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [coordinates[i]]
        
        clusters.append(current_cluster)
        return clusters
    
    def _validate_columns(self, column_boundaries: List[Tuple[float, float]], regions: List[OCRRegion]) -> List[Tuple[float, float]]:
        """Validate detected columns based on content distribution."""
        valid_columns = []
        
        for left, right in column_boundaries:
            # Count regions that fall within this column
            regions_in_column = 0
            total_text_length = 0
            
            for region in regions:
                bbox = region.bounding_box
                # Check if region center is within column boundaries
                center_x = (bbox.x1 + bbox.x2) / 2
                if left <= center_x <= right:
                    regions_in_column += 1
                    total_text_length += len(region.text)
            
            # Validate column based on content
            if regions_in_column >= 2 and total_text_length >= 50:  # Minimum content thresholds
                valid_columns.append((left, right))
        
        # If no valid columns found, assume single column
        if not valid_columns and regions:
            min_x = min(region.bounding_box.x1 for region in regions)
            max_x = max(region.bounding_box.x2 for region in regions)
            valid_columns = [(min_x, max_x)]
        
        return valid_columns
    
    def _calculate_column_gaps(self, columns: List[Tuple[float, float]]) -> List[float]:
        """Calculate gaps between columns."""
        if len(columns) < 2:
            return []
        
        gaps = []
        for i in range(len(columns) - 1):
            gap = columns[i+1][0] - columns[i][1]
            gaps.append(gap)
        
        return gaps
    
    def _assign_column_info(self, regions: List[OCRRegion], column_info: Dict[str, Any]):
        """Assign column information to regions."""
        column_boundaries = column_info.get('column_boundaries', [])
        
        for region in regions:
            bbox = region.bounding_box
            center_x = (bbox.x1 + bbox.x2) / 2
            
            # Find which column this region belongs to
            column_index = 0
            for i, (left, right) in enumerate(column_boundaries):
                if left <= center_x <= right:
                    column_index = i
                    break
            
            # Store column info in region (if we extended the model)
            # For now, we could use custom metadata
            # region.column_index = column_index
    
    def _determine_reading_order(self, regions: List[OCRRegion]):
        """Determine reading order for regions."""
        if not regions:
            return
        
        # Sort regions by position (top to bottom, left to right)
        # For multi-column layouts, this needs to be more sophisticated
        
        # Simple approach: sort by y-position first, then x-position
        sorted_regions = sorted(regions, key=lambda r: (r.bounding_box.y1, r.bounding_box.x1))
        
        # Assign reading order
        for i, region in enumerate(sorted_regions):
            region.reading_order = i
        
        logger.debug(f"Assigned reading order to {len(regions)} regions")
    
    def _detect_hierarchy(self, regions: List[OCRRegion]):
        """Detect hierarchical structure in regions."""
        if not regions:
            return
        
        # Enhance region type detection based on position and formatting
        for region in regions:
            region.region_type = self._refine_region_type(region, regions)
        
        # Establish parent-child relationships
        self._establish_hierarchy_relationships(regions)
    
    def _refine_region_type(self, region: OCRRegion, all_regions: List[OCRRegion]) -> RegionType:
        """Refine region type based on layout analysis."""
        bbox = region.bounding_box
        text = region.text.strip()
        
        # Get document dimensions
        if all_regions:
            doc_width = max(r.bounding_box.x2 for r in all_regions)
            doc_height = max(r.bounding_box.y2 for r in all_regions)
        else:
            doc_width = doc_height = 1000  # Default
        
        # Header detection (top 20% of page)
        if bbox.y1 < doc_height * 0.2:
            if len(text) < 100 and not text.endswith('.'):
                return RegionType.HEADER
        
        # Footer detection (bottom 10% of page)
        if bbox.y1 > doc_height * 0.9:
            return RegionType.FOOTER
        
        # Title detection (short text, potentially centered)
        if len(text) < 80 and not text.endswith('.'):
            # Check if roughly centered
            center_x = (bbox.x1 + bbox.x2) / 2
            if abs(center_x - doc_width / 2) < doc_width * 0.1:
                return RegionType.TITLE
        
        # List detection
        text_lower = text.lower()
        if (text_lower.startswith(('•', '·', '-', '*', '○', '■')) or
            (len(text) > 2 and text[0].isdigit() and text[1] in '.):')):
            return RegionType.LIST
        
        # Caption detection
        caption_keywords = ['figure', 'table', 'chart', 'graph', 'image', 'fig.', 'tab.']
        if any(keyword in text_lower for keyword in caption_keywords):
            return RegionType.CAPTION
        
        # Default to current type or paragraph
        return region.region_type if region.region_type != RegionType.UNKNOWN else RegionType.PARAGRAPH
    
    def _establish_hierarchy_relationships(self, regions: List[OCRRegion]):
        """Establish parent-child relationships between regions."""
        # Simple hierarchy based on region types and positions
        headers = [r for r in regions if r.region_type in [RegionType.HEADER, RegionType.TITLE]]
        
        for header in headers:
            # Find regions that come after this header and before the next header
            header_y = header.bounding_box.y1
            
            # Find next header
            next_headers = [h for h in headers if h.bounding_box.y1 > header_y]
            next_header_y = min(h.bounding_box.y1 for h in next_headers) if next_headers else float('inf')
            
            # Assign child regions
            for region in regions:
                if (header_y < region.bounding_box.y1 < next_header_y and
                    region.region_type not in [RegionType.HEADER, RegionType.TITLE]):
                    region.parent_region_id = header.region_id
                    header.child_region_ids.append(region.region_id)
    
    def get_layout_statistics(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Get layout analysis statistics."""
        regions = ocr_result.regions
        
        if not regions:
            return {}
        
        # Column detection
        column_info = self._detect_columns(regions)
        
        # Region type distribution
        type_distribution = {}
        for region in regions:
            region_type = region.region_type.value
            type_distribution[region_type] = type_distribution.get(region_type, 0) + 1
        
        # Reading order coverage
        regions_with_order = sum(1 for r in regions if r.reading_order is not None)
        
        # Hierarchical relationships
        regions_with_parents = sum(1 for r in regions if r.parent_region_id)
        regions_with_children = sum(1 for r in regions if r.child_region_ids)
        
        return {
            'total_regions': len(regions),
            'columns_detected': column_info.get('columns', 1),
            'column_boundaries': column_info.get('column_boundaries', []),
            'region_type_distribution': type_distribution,
            'reading_order_coverage': regions_with_order / len(regions),
            'hierarchical_structure': {
                'regions_with_parents': regions_with_parents,
                'regions_with_children': regions_with_children,
                'hierarchy_coverage': regions_with_parents / len(regions)
            }
        }
