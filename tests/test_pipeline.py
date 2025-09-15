"""
Tests for the main tax document processing pipeline.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from taxdoc_ai.pipeline.tax_document_processor import TaxDocumentProcessor
from taxdoc_ai.scoring.confidence_scorer import ExtractionResult

class TestTaxDocumentProcessor:
    """Test cases for TaxDocumentProcessor."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = TaxDocumentProcessor()
        assert processor is not None
        assert processor.classifier is not None
        assert processor.ocr is not None
        assert processor.ner_extractor is not None
        assert processor.confidence_scorer is not None
    
    def test_initialization_with_nvidia_key(self):
        """Test initialization with NVIDIA API key."""
        with patch('taxdoc_ai.ocr.ocr_processors.NVIDIAOCRProcessor') as mock_nvidia:
            processor = TaxDocumentProcessor(nvidia_api_key="test_key")
            # Should attempt to use NVIDIA processor
            assert processor is not None
    
    def test_initialization_with_model_paths(self):
        """Test initialization with model paths."""
        with patch('taxdoc_ai.classifiers.document_classifier.TaxDocumentClassifier.load_model') as mock_load:
            mock_load.return_value = Mock()
            processor = TaxDocumentProcessor(classifier_model_path="test_path.pth")
            assert processor is not None
    
    @patch('taxdoc_ai.pipeline.tax_document_processor.OCRProcessorFactory')
    def test_ocr_processor_initialization(self, mock_factory):
        """Test OCR processor initialization."""
        mock_processor = Mock()
        mock_factory.create_with_fallback.return_value = mock_processor
        
        processor = TaxDocumentProcessor()
        assert processor.ocr == mock_processor
    
    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        processor = TaxDocumentProcessor()
        stats = processor.get_processing_stats()
        
        assert 'classifier_info' in stats
        assert 'ocr_processor' in stats
        assert 'confidence_weights' in stats
    
    @patch('os.path.exists')
    def test_process_document_file_not_found(self, mock_exists):
        """Test processing non-existent document."""
        mock_exists.return_value = False
        processor = TaxDocumentProcessor()
        
        with pytest.raises(FileNotFoundError):
            processor.process_document("nonexistent.jpg")
    
    @patch('taxdoc_ai.pipeline.tax_document_processor.time.time')
    @patch('taxdoc_ai.pipeline.tax_document_processor.datetime')
    def test_process_document_success(self, mock_datetime, mock_time):
        """Test successful document processing."""
        # Mock time functions
        mock_time.side_effect = [0, 2.5]  # Start and end times
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-15T10:30:00"
        
        # Create processor with mocked components
        processor = TaxDocumentProcessor()
        
        # Mock OCR
        processor.ocr.extract_text_with_coordinates = Mock(return_value=[
            {'text': 'Sample text', 'confidence': 0.9, 'bbox': [[0, 0], [100, 0], [100, 20], [0, 20]]}
        ])
        
        # Mock classifier
        processor.classifier.predict = Mock(return_value="W-2")
        processor.classifier.get_confidence_score = Mock(return_value=0.95)
        
        # Mock NER extractor
        processor.ner_extractor.extract_tax_entities = Mock(return_value={
            'SSN': [{'text': '123-45-6789', 'confidence': 0.9, 'start': 0, 'end': 11}]
        })
        
        # Mock confidence scorer
        processor.confidence_scorer.score_extraction_batch = Mock(return_value=[
            ExtractionResult(
                field_name='SSN',
                value='123-45-6789',
                confidence=0.9,
                source_confidence=0.9,
                model_confidence=0.9,
                validation_score=0.9
            )
        ])
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(b'fake image data')
            tmp_path = tmp_file.name
        
        try:
            result = processor.process_document(tmp_path)
            
            assert result['document_type'] == "W-2"
            assert result['classification_confidence'] == 0.95
            assert len(result['extracted_fields']) == 1
            assert result['extracted_fields'][0]['field_name'] == 'SSN'
            assert result['processing_metadata']['processing_time_seconds'] == 2.5
            
        finally:
            os.unlink(tmp_path)
    
    def test_process_batch(self):
        """Test batch processing."""
        processor = TaxDocumentProcessor()
        
        # Mock process_document method
        processor.process_document = Mock(return_value={
            'document_type': 'W-2',
            'classification_confidence': 0.95,
            'extracted_fields': [],
            'raw_text': 'Sample text',
            'processing_metadata': {'processing_time_seconds': 2.5},
            'confidence_summary': {'average_confidence': 0.95}
        })
        
        # Create temporary files
        temp_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(b'fake image data')
                temp_files.append(tmp_file.name)
        
        try:
            results = processor.process_batch(temp_files)
            
            assert len(results) == 3
            for result in results:
                assert result['document_type'] == 'W-2'
                assert result['classification_confidence'] == 0.95
                
        finally:
            for tmp_file in temp_files:
                os.unlink(tmp_file)
    
    def test_get_structured_data(self):
        """Test getting structured data."""
        processor = TaxDocumentProcessor()
        
        # Mock process_document method
        processor.process_document = Mock(return_value={
            'extracted_fields': [
                {'field_name': 'SSN', 'value': '123-45-6789', 'confidence': 0.9},
                {'field_name': 'WAGE_AMOUNT', 'value': '$50,000.00', 'confidence': 0.8},
                {'field_name': 'EMPLOYER_NAME', 'value': 'ABC Corp', 'confidence': 0.3}  # Low confidence
            ]
        })
        
        structured_data = processor.get_structured_data("test.jpg")
        
        assert structured_data['SSN'] == '123-45-6789'
        assert structured_data['WAGE_AMOUNT'] == '$50,000.00'
        assert structured_data['EMPLOYER_NAME'] is None  # Low confidence filtered out
    
    def test_validate_document(self):
        """Test document validation."""
        processor = TaxDocumentProcessor()
        
        # Mock process_document method
        processor.process_document = Mock(return_value={
            'document_type': 'W-2',
            'classification_confidence': 0.95,
            'extracted_fields': [
                {'field_name': 'SSN', 'value': '123-45-6789', 'confidence': 0.9},
                {'field_name': 'WAGE_AMOUNT', 'value': '$50,000.00', 'confidence': 0.8}
            ],
            'confidence_summary': {'average_confidence': 0.85},
            'processing_metadata': {'processing_time_seconds': 2.5}
        })
        
        validation = processor.validate_document("test.jpg", expected_type="W-2")
        
        assert validation['is_valid'] is True
        assert validation['confidence_score'] == 0.85
        assert len(validation['issues']) == 0
    
    def test_validate_document_wrong_type(self):
        """Test document validation with wrong expected type."""
        processor = TaxDocumentProcessor()
        
        # Mock process_document method
        processor.process_document = Mock(return_value={
            'document_type': 'W-2',
            'classification_confidence': 0.95,
            'extracted_fields': [],
            'confidence_summary': {'average_confidence': 0.85},
            'processing_metadata': {'processing_time_seconds': 2.5}
        })
        
        validation = processor.validate_document("test.jpg", expected_type="1099-R")
        
        assert validation['is_valid'] is True  # No critical issues
        assert len(validation['warnings']) > 0  # Should have type mismatch warning
    
    def test_validate_document_low_confidence(self):
        """Test document validation with low confidence."""
        processor = TaxDocumentProcessor()
        
        # Mock process_document method
        processor.process_document = Mock(return_value={
            'document_type': 'W-2',
            'classification_confidence': 0.95,
            'extracted_fields': [],
            'confidence_summary': {'average_confidence': 0.3},  # Low confidence
            'processing_metadata': {'processing_time_seconds': 2.5}
        })
        
        validation = processor.validate_document("test.jpg")
        
        assert validation['is_valid'] is False
        assert len(validation['issues']) > 0  # Should have low confidence issue
