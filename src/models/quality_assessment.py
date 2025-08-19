"""
Quality assessment models for evaluating OCR results and processing quality.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class QualityLevel(str, Enum):
    """Quality levels for OCR assessment."""
    EXCELLENT = "excellent"  # >= 0.9
    GOOD = "good"           # >= 0.8
    ACCEPTABLE = "acceptable"  # >= 0.7
    POOR = "poor"           # >= 0.5
    UNUSABLE = "unusable"   # < 0.5


class QualityIssue(str, Enum):
    """Types of quality issues that can be detected."""
    LOW_CONFIDENCE = "low_confidence"
    GARBLED_TEXT = "garbled_text"
    MISSING_TEXT = "missing_text"
    LAYOUT_ERRORS = "layout_errors"
    LANGUAGE_DETECTION_FAILED = "language_detection_failed"
    TABLE_EXTRACTION_FAILED = "table_extraction_failed"
    FIGURE_DETECTION_FAILED = "figure_detection_failed"
    EXCESSIVE_NOISE = "excessive_noise"
    POOR_IMAGE_QUALITY = "poor_image_quality"
    ROTATION_ISSUES = "rotation_issues"
    SCALING_ISSUES = "scaling_issues"


class ConfidenceMetrics(BaseModel):
    """Detailed confidence score metrics."""
    mean: float = Field(..., description="Mean confidence score")
    median: float = Field(..., description="Median confidence score")
    std_dev: float = Field(..., description="Standard deviation of confidence scores")
    min_value: float = Field(..., description="Minimum confidence score")
    max_value: float = Field(..., description="Maximum confidence score")
    
    # Percentile values
    percentile_25: float = Field(..., description="25th percentile confidence")
    percentile_75: float = Field(..., description="75th percentile confidence")
    percentile_90: float = Field(..., description="90th percentile confidence")
    
    # Distribution analysis
    low_confidence_count: int = Field(default=0, description="Count of regions with confidence < 0.5")
    medium_confidence_count: int = Field(default=0, description="Count of regions with 0.5 <= confidence < 0.8")
    high_confidence_count: int = Field(default=0, description="Count of regions with confidence >= 0.8")
    
    @validator('mean', 'median', 'min_value', 'max_value', 'percentile_25', 'percentile_75', 'percentile_90')
    def validate_confidence_values(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence values must be between 0.0 and 1.0')
        return v
    
    @property
    def quality_level(self) -> QualityLevel:
        """Determine quality level based on mean confidence."""
        if self.mean >= 0.9:
            return QualityLevel.EXCELLENT
        elif self.mean >= 0.8:
            return QualityLevel.GOOD
        elif self.mean >= 0.7:
            return QualityLevel.ACCEPTABLE
        elif self.mean >= 0.5:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE
    
    @property
    def is_consistent(self) -> bool:
        """Check if confidence scores are consistent (low variance)."""
        return self.std_dev < 0.2
    
    @property
    def high_confidence_ratio(self) -> float:
        """Calculate ratio of high-confidence regions."""
        total = self.low_confidence_count + self.medium_confidence_count + self.high_confidence_count
        if total > 0:
            return self.high_confidence_count / total
        return 0.0


class ContentQualityMetrics(BaseModel):
    """Metrics for assessing content quality."""
    # Text characteristics
    total_characters: int = Field(default=0, description="Total characters extracted")
    total_words: int = Field(default=0, description="Total words extracted")
    average_word_length: float = Field(default=0.0, description="Average word length")
    
    # Language and readability
    primary_language: Optional[str] = Field(None, description="Primary detected language")
    language_confidence: float = Field(default=0.0, description="Language detection confidence")
    readability_score: Optional[float] = Field(None, description="Text readability score")
    
    # Content structure
    sentences_detected: int = Field(default=0, description="Number of sentences detected")
    paragraphs_detected: int = Field(default=0, description="Number of paragraphs detected")
    headers_detected: int = Field(default=0, description="Number of headers detected")
    
    # Special content
    tables_detected: int = Field(default=0, description="Number of tables detected")
    figures_detected: int = Field(default=0, description="Number of figures detected")
    lists_detected: int = Field(default=0, description="Number of lists detected")
    
    # Quality indicators
    special_character_ratio: float = Field(default=0.0, description="Ratio of special/non-alphanumeric characters")
    digit_ratio: float = Field(default=0.0, description="Ratio of numeric characters")
    uppercase_ratio: float = Field(default=0.0, description="Ratio of uppercase characters")
    
    # Potential issues
    repeated_character_sequences: int = Field(default=0, description="Count of likely OCR artifacts")
    very_short_words: int = Field(default=0, description="Count of single-character 'words'")
    very_long_words: int = Field(default=0, description="Count of unusually long words (>20 chars)")
    
    @property
    def average_words_per_sentence(self) -> float:
        """Calculate average words per sentence."""
        if self.sentences_detected > 0:
            return self.total_words / self.sentences_detected
        return 0.0
    
    @property
    def has_reasonable_structure(self) -> bool:
        """Check if content has reasonable structure."""
        return (self.sentences_detected > 0 and 
                self.average_words_per_sentence > 3 and
                self.average_word_length > 2)
    
    @property
    def artifact_ratio(self) -> float:
        """Calculate ratio of likely OCR artifacts."""
        total_issues = self.repeated_character_sequences + self.very_short_words
        if self.total_words > 0:
            return total_issues / self.total_words
        return 0.0


class LayoutQualityMetrics(BaseModel):
    """Metrics for assessing layout detection quality."""
    # Spatial coverage
    text_coverage_ratio: float = Field(..., description="Ratio of image area covered by text")
    layout_consistency_score: float = Field(default=0.0, description="Layout consistency score")
    
    # Region detection
    total_regions_detected: int = Field(default=0, description="Total text regions detected")
    overlapping_regions: int = Field(default=0, description="Number of overlapping regions")
    orphaned_regions: int = Field(default=0, description="Regions with no clear context")
    
    # Reading order
    reading_order_detected: bool = Field(default=False, description="Reading order successfully detected")
    reading_order_confidence: float = Field(default=0.0, description="Confidence in reading order")
    
    # Column and structure detection
    columns_detected: int = Field(default=1, description="Number of columns detected")
    column_layout_score: float = Field(default=0.0, description="Column layout quality score")
    
    # Hierarchical structure
    hierarchy_levels_detected: int = Field(default=0, description="Number of hierarchy levels")
    title_detection_confidence: float = Field(default=0.0, description="Title detection confidence")
    
    @validator('text_coverage_ratio', 'layout_consistency_score', 'reading_order_confidence', 
               'column_layout_score', 'title_detection_confidence')
    def validate_ratio_values(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Ratio/score values must be between 0.0 and 1.0')
        return v
    
    @property
    def has_good_layout_detection(self) -> bool:
        """Check if layout detection is good quality."""
        return (self.text_coverage_ratio > 0.1 and
                self.layout_consistency_score > 0.7 and
                self.overlapping_regions < self.total_regions_detected * 0.1)


class QualityAssessment(BaseModel):
    """
    Comprehensive quality assessment for OCR processing results.
    
    Provides detailed analysis of OCR quality, content quality, and layout detection.
    """
    # Identification
    document_id: str = Field(..., description="Assessed document ID")
    page_number: Optional[int] = Field(None, description="Page number (if page-specific)")
    assessment_id: str = Field(..., description="Unique assessment ID")
    
    # Overall scores
    overall_quality_score: float = Field(..., description="Overall quality score (0-1)")
    overall_quality_level: QualityLevel = Field(..., description="Overall quality level")
    is_acceptable_for_rag: bool = Field(..., description="Whether suitable for RAG usage")
    
    # Detailed metrics
    confidence_metrics: ConfidenceMetrics = Field(..., description="OCR confidence analysis")
    content_metrics: ContentQualityMetrics = Field(..., description="Content quality analysis")
    layout_metrics: LayoutQualityMetrics = Field(..., description="Layout quality analysis")
    
    # Issues and recommendations
    detected_issues: List[QualityIssue] = Field(default_factory=list, description="Quality issues detected")
    issue_descriptions: Dict[QualityIssue, str] = Field(default_factory=dict, description="Detailed issue descriptions")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    # Processing context
    ocr_engine_used: str = Field(..., description="OCR engine used for processing")
    preprocessing_applied: List[str] = Field(default_factory=list, description="Preprocessing steps applied")
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Assessment timestamp")
    
    # Performance impact
    processing_time_seconds: float = Field(default=0.0, description="Time taken for assessment")
    
    @validator('overall_quality_score')
    def validate_overall_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Overall quality score must be between 0.0 and 1.0')
        return v
    
    @property
    def quality_summary(self) -> Dict[str, Any]:
        """Get concise quality summary."""
        return {
            "overall_score": round(self.overall_quality_score, 2),
            "quality_level": self.overall_quality_level.value,
            "confidence_mean": round(self.confidence_metrics.mean, 2),
            "text_coverage": round(self.layout_metrics.text_coverage_ratio, 2),
            "total_words": self.content_metrics.total_words,
            "issues_count": len(self.detected_issues),
            "suitable_for_rag": self.is_acceptable_for_rag
        }
    
    @property
    def major_issues(self) -> List[QualityIssue]:
        """Get major quality issues that significantly impact usability."""
        major_issue_types = {
            QualityIssue.LOW_CONFIDENCE,
            QualityIssue.GARBLED_TEXT,
            QualityIssue.MISSING_TEXT,
            QualityIssue.POOR_IMAGE_QUALITY
        }
        return [issue for issue in self.detected_issues if issue in major_issue_types]
    
    @property
    def needs_reprocessing(self) -> bool:
        """Check if document should be reprocessed with different settings."""
        return (self.overall_quality_score < 0.6 or 
                len(self.major_issues) > 0 or
                self.confidence_metrics.high_confidence_ratio < 0.5)
    
    def add_issue(self, issue: QualityIssue, description: str):
        """Add a quality issue with description."""
        if issue not in self.detected_issues:
            self.detected_issues.append(issue)
        self.issue_descriptions[issue] = description
    
    def add_recommendation(self, recommendation: str):
        """Add an improvement recommendation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed quality assessment report."""
        return {
            "document_id": self.document_id,
            "assessment_summary": self.quality_summary,
            "confidence_analysis": {
                "mean_confidence": self.confidence_metrics.mean,
                "confidence_distribution": {
                    "high": self.confidence_metrics.high_confidence_count,
                    "medium": self.confidence_metrics.medium_confidence_count,
                    "low": self.confidence_metrics.low_confidence_count
                },
                "consistency": "consistent" if self.confidence_metrics.is_consistent else "variable"
            },
            "content_analysis": {
                "total_words": self.content_metrics.total_words,
                "structure_quality": "good" if self.content_metrics.has_reasonable_structure else "poor",
                "language": self.content_metrics.primary_language,
                "special_content": {
                    "tables": self.content_metrics.tables_detected,
                    "figures": self.content_metrics.figures_detected,
                    "headers": self.content_metrics.headers_detected
                }
            },
            "layout_analysis": {
                "text_coverage": self.layout_metrics.text_coverage_ratio,
                "layout_quality": "good" if self.layout_metrics.has_good_layout_detection else "poor",
                "columns": self.layout_metrics.columns_detected,
                "reading_order": "detected" if self.layout_metrics.reading_order_detected else "not_detected"
            },
            "issues": {
                "total_issues": len(self.detected_issues),
                "major_issues": [issue.value for issue in self.major_issues],
                "issue_details": {issue.value: desc for issue, desc in self.issue_descriptions.items()}
            },
            "recommendations": self.recommendations,
            "verdict": {
                "suitable_for_rag": self.is_acceptable_for_rag,
                "needs_reprocessing": self.needs_reprocessing,
                "quality_level": self.overall_quality_level.value
            }
        }
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
