"""
OCR quality assessor for evaluating OCR results and providing recommendations.
"""

import uuid
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from ..models import (
    OCRResult, QualityAssessment, ConfidenceMetrics, ContentQualityMetrics,
    LayoutQualityMetrics, QualityLevel, QualityIssue
)

logger = logging.getLogger(__name__)


class OCRQualityAssessor:
    """
    Comprehensive OCR quality assessment system.
    
    Evaluates OCR results across multiple dimensions including confidence,
    content quality, layout detection, and provides actionable recommendations.
    """
    
    def __init__(self,
                 min_confidence_threshold: float = 0.6,
                 enable_content_analysis: bool = True,
                 enable_layout_analysis: bool = True,
                 language_detection_enabled: bool = True):
        """
        Initialize OCR quality assessor.
        
        Args:
            min_confidence_threshold: Minimum acceptable confidence
            enable_content_analysis: Enable content quality analysis
            enable_layout_analysis: Enable layout quality analysis
            language_detection_enabled: Enable language detection analysis
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_content_analysis = enable_content_analysis
        self.enable_layout_analysis = enable_layout_analysis
        self.language_detection_enabled = language_detection_enabled
        
        logger.info("OCR Quality Assessor initialized")
    
    async def assess_ocr_result(self, ocr_result: OCRResult) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of OCR result.
        
        Args:
            ocr_result: OCR result to assess
            
        Returns:
            Quality assessment with scores and recommendations
        """
        start_time = datetime.utcnow()
        assessment_id = str(uuid.uuid4())
        
        logger.info(f"Starting quality assessment for document {ocr_result.document_id}, page {ocr_result.page_number}")
        
        # Analyze confidence metrics
        confidence_metrics = self._analyze_confidence_metrics(ocr_result)
        
        # Analyze content quality
        content_metrics = None
        if self.enable_content_analysis:
            content_metrics = self._analyze_content_quality(ocr_result)
        else:
            content_metrics = ContentQualityMetrics()
        
        # Analyze layout quality
        layout_metrics = None
        if self.enable_layout_analysis:
            layout_metrics = self._analyze_layout_quality(ocr_result)
        else:
            layout_metrics = LayoutQualityMetrics(text_coverage_ratio=ocr_result.text_coverage_ratio)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(
            confidence_metrics, content_metrics, layout_metrics
        )
        
        # Determine quality level
        quality_level = self._determine_quality_level(overall_score)
        
        # Detect issues and generate recommendations
        detected_issues = []
        recommendations = []
        issue_descriptions = {}
        
        self._detect_confidence_issues(confidence_metrics, detected_issues, issue_descriptions, recommendations)
        
        if self.enable_content_analysis:
            self._detect_content_issues(content_metrics, detected_issues, issue_descriptions, recommendations)
        
        if self.enable_layout_analysis:
            self._detect_layout_issues(layout_metrics, detected_issues, issue_descriptions, recommendations)
        
        # Determine if suitable for RAG
        is_acceptable = self._is_acceptable_for_rag(overall_score, detected_issues)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        assessment = QualityAssessment(
            document_id=ocr_result.document_id,
            page_number=ocr_result.page_number,
            assessment_id=assessment_id,
            overall_quality_score=overall_score,
            overall_quality_level=quality_level,
            is_acceptable_for_rag=is_acceptable,
            confidence_metrics=confidence_metrics,
            content_metrics=content_metrics,
            layout_metrics=layout_metrics,
            detected_issues=detected_issues,
            issue_descriptions=issue_descriptions,
            recommendations=recommendations,
            ocr_engine_used=ocr_result.ocr_engine.value,
            preprocessing_applied=ocr_result.preprocessing_applied,
            processing_time_seconds=processing_time
        )
        
        logger.info(f"Quality assessment completed: score={overall_score:.2f}, level={quality_level.value}")
        return assessment
    
    def _analyze_confidence_metrics(self, ocr_result: OCRResult) -> ConfidenceMetrics:
        """Analyze confidence scores from OCR result."""
        if not ocr_result.regions:
            return ConfidenceMetrics(
                mean=0.0, median=0.0, std_dev=0.0,
                min_value=0.0, max_value=0.0,
                percentile_25=0.0, percentile_75=0.0, percentile_90=0.0
            )
        
        confidences = [region.confidence for region in ocr_result.regions]
        
        # Calculate basic statistics
        mean_conf = statistics.mean(confidences)
        median_conf = statistics.median(confidences)
        std_dev = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        min_conf = min(confidences)
        max_conf = max(confidences)
        
        # Calculate percentiles
        sorted_conf = sorted(confidences)
        n = len(sorted_conf)
        percentile_25 = sorted_conf[int(n * 0.25)] if n > 0 else 0.0
        percentile_75 = sorted_conf[int(n * 0.75)] if n > 0 else 0.0
        percentile_90 = sorted_conf[int(n * 0.90)] if n > 0 else 0.0
        
        # Count confidence levels
        low_conf_count = sum(1 for c in confidences if c < 0.5)
        medium_conf_count = sum(1 for c in confidences if 0.5 <= c < 0.8)
        high_conf_count = sum(1 for c in confidences if c >= 0.8)
        
        return ConfidenceMetrics(
            mean=mean_conf,
            median=median_conf,
            std_dev=std_dev,
            min_value=min_conf,
            max_value=max_conf,
            percentile_25=percentile_25,
            percentile_75=percentile_75,
            percentile_90=percentile_90,
            low_confidence_count=low_conf_count,
            medium_confidence_count=medium_conf_count,
            high_confidence_count=high_conf_count
        )
    
    def _analyze_content_quality(self, ocr_result: OCRResult) -> ContentQualityMetrics:
        """Analyze content quality metrics."""
        all_text = ocr_result.total_text
        
        if not all_text.strip():
            return ContentQualityMetrics()
        
        # Basic text statistics
        total_chars = len(all_text)
        words = all_text.split()
        total_words = len(words)
        avg_word_length = sum(len(word) for word in words) / total_words if words else 0.0
        
        # Language detection
        primary_language = self._detect_language(all_text) if self.language_detection_enabled else None
        language_confidence = 0.8 if primary_language else 0.0  # Simplified
        
        # Content structure analysis
        sentences = self._count_sentences(all_text)
        paragraphs = len([r for r in ocr_result.regions if r.region_type.value == 'paragraph'])
        headers = len([r for r in ocr_result.regions if r.region_type.value in ['header', 'title']])
        
        # Special content
        tables = len(ocr_result.tables)
        figures = len([r for r in ocr_result.regions if r.region_type.value == 'figure'])
        lists = len([r for r in ocr_result.regions if r.region_type.value == 'list'])
        
        # Character analysis
        special_chars = sum(1 for c in all_text if not c.isalnum() and not c.isspace())
        digits = sum(1 for c in all_text if c.isdigit())
        uppercase = sum(1 for c in all_text if c.isupper())
        
        special_char_ratio = special_chars / total_chars if total_chars > 0 else 0.0
        digit_ratio = digits / total_chars if total_chars > 0 else 0.0
        uppercase_ratio = uppercase / total_chars if total_chars > 0 else 0.0
        
        # OCR artifact detection
        repeated_sequences = self._detect_repeated_sequences(all_text)
        very_short_words = sum(1 for word in words if len(word.strip()) == 1)
        very_long_words = sum(1 for word in words if len(word.strip()) > 20)
        
        return ContentQualityMetrics(
            total_characters=total_chars,
            total_words=total_words,
            average_word_length=avg_word_length,
            primary_language=primary_language,
            language_confidence=language_confidence,
            sentences_detected=sentences,
            paragraphs_detected=paragraphs,
            headers_detected=headers,
            tables_detected=tables,
            figures_detected=figures,
            lists_detected=lists,
            special_character_ratio=special_char_ratio,
            digit_ratio=digit_ratio,
            uppercase_ratio=uppercase_ratio,
            repeated_character_sequences=repeated_sequences,
            very_short_words=very_short_words,
            very_long_words=very_long_words
        )
    
    def _analyze_layout_quality(self, ocr_result: OCRResult) -> LayoutQualityMetrics:
        """Analyze layout detection quality."""
        # Basic layout metrics
        text_coverage = ocr_result.text_coverage_ratio
        total_regions = len(ocr_result.regions)
        
        # Detect overlapping regions
        overlapping = self._count_overlapping_regions(ocr_result.regions)
        
        # Reading order analysis
        reading_order_detected = any(r.reading_order is not None for r in ocr_result.regions)
        reading_order_confidence = 0.8 if reading_order_detected else 0.0
        
        # Column detection (simplified)
        columns = self._estimate_column_count(ocr_result.regions)
        
        # Hierarchy detection
        hierarchy_levels = len(set(r.region_type for r in ocr_result.regions))
        title_regions = [r for r in ocr_result.regions if r.region_type.value in ['title', 'header']]
        title_confidence = 0.8 if title_regions else 0.0
        
        # Layout consistency (simplified metric)
        layout_consistency = self._calculate_layout_consistency(ocr_result.regions)
        
        return LayoutQualityMetrics(
            text_coverage_ratio=text_coverage,
            layout_consistency_score=layout_consistency,
            total_regions_detected=total_regions,
            overlapping_regions=overlapping,
            reading_order_detected=reading_order_detected,
            reading_order_confidence=reading_order_confidence,
            columns_detected=columns,
            hierarchy_levels_detected=hierarchy_levels,
            title_detection_confidence=title_confidence
        )
    
    def _calculate_overall_quality_score(self,
                                       confidence_metrics: ConfidenceMetrics,
                                       content_metrics: ContentQualityMetrics,
                                       layout_metrics: LayoutQualityMetrics) -> float:
        """Calculate overall quality score from component metrics."""
        # Weighted combination of quality factors
        confidence_score = confidence_metrics.mean
        
        # Content quality score
        content_score = 0.5  # Default
        if content_metrics.has_reasonable_structure:
            content_score += 0.3
        if content_metrics.artifact_ratio < 0.1:
            content_score += 0.2
        content_score = min(content_score, 1.0)
        
        # Layout quality score
        layout_score = 0.5  # Default
        if layout_metrics.has_good_layout_detection:
            layout_score += 0.3
        if layout_metrics.text_coverage_ratio > 0.1:
            layout_score += 0.2
        layout_score = min(layout_score, 1.0)
        
        # Weighted average
        weights = [0.5, 0.3, 0.2]  # Confidence, content, layout
        scores = [confidence_score, content_score, layout_score]
        
        overall_score = sum(w * s for w, s in zip(weights, scores))
        return min(max(overall_score, 0.0), 1.0)
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from overall score."""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.8:
            return QualityLevel.GOOD
        elif score >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE
    
    def _is_acceptable_for_rag(self, score: float, issues: List[QualityIssue]) -> bool:
        """Determine if quality is acceptable for RAG usage."""
        # Must meet minimum score threshold
        if score < 0.6:
            return False
        
        # Check for critical issues
        critical_issues = {
            QualityIssue.GARBLED_TEXT,
            QualityIssue.MISSING_TEXT,
            QualityIssue.POOR_IMAGE_QUALITY
        }
        
        if any(issue in critical_issues for issue in issues):
            return False
        
        return True
    
    def _detect_confidence_issues(self, metrics: ConfidenceMetrics, 
                                issues: List[QualityIssue], 
                                descriptions: Dict[QualityIssue, str],
                                recommendations: List[str]):
        """Detect confidence-related quality issues."""
        
        if metrics.mean < 0.6:
            issues.append(QualityIssue.LOW_CONFIDENCE)
            descriptions[QualityIssue.LOW_CONFIDENCE] = f"Average confidence {metrics.mean:.2f} below threshold"
            recommendations.append("Consider reprocessing with different OCR engine or settings")
        
        if metrics.std_dev > 0.3:
            descriptions[QualityIssue.LOW_CONFIDENCE] = descriptions.get(
                QualityIssue.LOW_CONFIDENCE, ""
            ) + f" High confidence variance: {metrics.std_dev:.2f}"
            recommendations.append("High confidence variation detected - check image quality")
    
    def _detect_content_issues(self, metrics: ContentQualityMetrics,
                             issues: List[QualityIssue],
                             descriptions: Dict[QualityIssue, str],
                             recommendations: List[str]):
        """Detect content-related quality issues."""
        
        if metrics.artifact_ratio > 0.2:
            issues.append(QualityIssue.GARBLED_TEXT)
            descriptions[QualityIssue.GARBLED_TEXT] = f"High OCR artifact ratio: {metrics.artifact_ratio:.2f}"
            recommendations.append("Consider image preprocessing to reduce noise")
        
        if not metrics.has_reasonable_structure:
            issues.append(QualityIssue.LAYOUT_ERRORS)
            descriptions[QualityIssue.LAYOUT_ERRORS] = "Poor text structure detected"
            recommendations.append("Consider using layout-aware OCR settings")
        
        if metrics.language_confidence < 0.5:
            issues.append(QualityIssue.LANGUAGE_DETECTION_FAILED)
            descriptions[QualityIssue.LANGUAGE_DETECTION_FAILED] = "Language detection uncertain"
            recommendations.append("Specify language explicitly for better OCR results")
    
    def _detect_layout_issues(self, metrics: LayoutQualityMetrics,
                            issues: List[QualityIssue],
                            descriptions: Dict[QualityIssue, str],
                            recommendations: List[str]):
        """Detect layout-related quality issues."""
        
        if metrics.text_coverage_ratio < 0.05:
            issues.append(QualityIssue.MISSING_TEXT)
            descriptions[QualityIssue.MISSING_TEXT] = f"Low text coverage: {metrics.text_coverage_ratio:.2f}"
            recommendations.append("Check if document contains primarily images or is blank")
        
        if metrics.overlapping_regions > metrics.total_regions_detected * 0.2:
            issues.append(QualityIssue.LAYOUT_ERRORS)
            descriptions[QualityIssue.LAYOUT_ERRORS] = f"Many overlapping regions: {metrics.overlapping_regions}"
            recommendations.append("Consider using different page segmentation mode")
    
    # Helper methods
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect primary language of text."""
        try:
            from langdetect import detect
            if len(text.strip()) > 20:
                return detect(text)
        except:
            pass
        return None
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        import re
        sentence_endings = re.findall(r'[.!?]+', text)
        return len(sentence_endings)
    
    def _detect_repeated_sequences(self, text: str) -> int:
        """Detect repeated character sequences (OCR artifacts)."""
        import re
        # Find sequences of 3+ repeated characters
        repeated = re.findall(r'(.)\1{2,}', text)
        return len(repeated)
    
    def _count_overlapping_regions(self, regions) -> int:
        """Count overlapping text regions."""
        overlapping = 0
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                if region1.bounding_box.overlaps_with(region2.bounding_box):
                    overlapping += 1
        return overlapping
    
    def _estimate_column_count(self, regions) -> int:
        """Estimate number of columns in document."""
        if not regions:
            return 1
        
        # Simple heuristic based on horizontal clustering
        x_positions = [region.bounding_box.x1 for region in regions]
        unique_positions = len(set(round(x, -1) for x in x_positions))  # Round to nearest 10
        return min(max(unique_positions, 1), 5)  # Cap at 5 columns
    
    def _calculate_layout_consistency(self, regions) -> float:
        """Calculate layout consistency score."""
        if not regions:
            return 0.0
        
        # Simple metric based on region alignment
        left_margins = [region.bounding_box.x1 for region in regions]
        margin_variance = statistics.variance(left_margins) if len(left_margins) > 1 else 0.0
        
        # Lower variance = higher consistency
        consistency = max(0.0, 1.0 - (margin_variance / 1000.0))  # Normalized
        return min(consistency, 1.0)
