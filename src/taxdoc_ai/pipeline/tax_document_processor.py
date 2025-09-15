"""
Main Tax Document Processor Pipeline.

This module implements the complete pipeline that orchestrates all components
for tax document processing: OCR, classification, NER, and confidence scoring.
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np

from ..classifiers.document_classifier import TaxDocumentClassifier
from ..ocr.ocr_processors import OCRProcessorFactory, BaseOCRProcessor
from ..ner.field_extractor import TaxFieldExtractor
from ..scoring.confidence_scorer import ConfidenceScorer, ExtractionResult

logger = logging.getLogger(__name__)

class TaxDocumentProcessor:
    """
    Main pipeline for processing tax documents.
    
    This class orchestrates the complete tax document processing pipeline:
    1. OCR text extraction
    2. Document classification
    3. Named entity extraction
    4. Confidence scoring
    """
    
    def __init__(
        self,
        nvidia_api_key: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        ner_model_path: Optional[str] = None,
        ocr_processor_type: str = "auto",
        confidence_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the TaxDocumentProcessor.
        
        Args:
            nvidia_api_key: NVIDIA API key for OCR (optional)
            classifier_model_path: Path to trained classifier model
            ner_model_path: Path to trained NER model
            ocr_processor_type: Type of OCR processor ("nvidia", "paddle", "auto")
            confidence_weights: Weights for confidence scoring
        """
        self.nvidia_api_key = nvidia_api_key
        self.classifier_model_path = classifier_model_path
        self.ner_model_path = ner_model_path
        self.ocr_processor_type = ocr_processor_type
        
        # Initialize components
        self._initialize_components(confidence_weights)
        
        logger.info("TaxDocumentProcessor initialized successfully")
    
    def _initialize_components(self, confidence_weights: Optional[Dict[str, float]] = None):
        """Initialize all pipeline components."""
        try:
            # Initialize document classifier
            self.classifier = self._initialize_classifier()
            
            # Initialize OCR processor
            self.ocr = self._initialize_ocr_processor()
            
            # Initialize NER extractor
            self.ner_extractor = self._initialize_ner_extractor()
            
            # Initialize confidence scorer
            self.confidence_scorer = ConfidenceScorer(weights=confidence_weights)
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_classifier(self) -> TaxDocumentClassifier:
        """Initialize the document classifier."""
        try:
            if self.classifier_model_path and os.path.exists(self.classifier_model_path):
                classifier = TaxDocumentClassifier.load_model(self.classifier_model_path)
                logger.info(f"Loaded trained classifier from {self.classifier_model_path}")
            else:
                classifier = TaxDocumentClassifier()
                logger.warning("Using untrained classifier - consider training for better performance")
            
            return classifier
            
        except Exception as e:
            logger.error(f"Failed to initialize classifier: {e}")
            raise
    
    def _initialize_ocr_processor(self) -> BaseOCRProcessor:
        """Initialize the OCR processor."""
        try:
            if self.ocr_processor_type == "auto":
                # Use NVIDIA if API key available, otherwise PaddleOCR
                processor = OCRProcessorFactory.create_with_fallback(
                    nvidia_api_key=self.nvidia_api_key
                )
            elif self.ocr_processor_type == "nvidia":
                if not self.nvidia_api_key:
                    raise ValueError("NVIDIA API key required for NVIDIA OCR processor")
                processor = OCRProcessorFactory.create_processor(
                    "nvidia", api_key=self.nvidia_api_key
                )
            elif self.ocr_processor_type == "paddle":
                processor = OCRProcessorFactory.create_processor("paddle")
            else:
                raise ValueError(f"Unknown OCR processor type: {self.ocr_processor_type}")
            
            logger.info(f"Initialized OCR processor: {type(processor).__name__}")
            return processor
            
        except Exception as e:
            logger.error(f"Failed to initialize OCR processor: {e}")
            raise
    
    def _initialize_ner_extractor(self) -> TaxFieldExtractor:
        """Initialize the NER extractor."""
        try:
            ner_extractor = TaxFieldExtractor(model_path=self.ner_model_path)
            logger.info("Initialized NER extractor")
            return ner_extractor
            
        except Exception as e:
            logger.error(f"Failed to initialize NER extractor: {e}")
            raise
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single tax document.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            logger.info(f"Processing document: {image_path}")
            
            # Step 1: OCR text extraction
            extracted_text, ocr_confidence = self._extract_text(image_path)
            
            # Step 2: Document classification
            doc_type, classification_confidence = self._classify_document(extracted_text)
            
            # Step 3: Named entity extraction
            tax_entities = self.ner_extractor.extract_tax_entities(extracted_text)
            
            # Step 4: Create extraction results and apply confidence scoring
            extraction_results = self._create_extraction_results(
                tax_entities, ocr_confidence
            )
            
            # Step 5: Apply confidence scoring
            scored_results = self.confidence_scorer.score_extraction_batch(extraction_results)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result dictionary
            result = {
                'document_type': doc_type,
                'classification_confidence': classification_confidence,
                'extracted_fields': [
                    {
                        'field_name': r.field_name,
                        'value': r.value,
                        'confidence': r.confidence,
                        'source_confidence': r.source_confidence,
                        'model_confidence': r.model_confidence,
                        'validation_score': r.validation_score,
                        'start_position': r.start_position,
                        'end_position': r.end_position
                    }
                    for r in scored_results
                ],
                'raw_text': extracted_text,
                'processing_metadata': {
                    'ocr_confidence': ocr_confidence,
                    'processing_time_seconds': processing_time,
                    'timestamp': datetime.now().isoformat(),
                    'ocr_processor': type(self.ocr).__name__,
                    'image_path': image_path
                },
                'confidence_summary': self.confidence_scorer.get_confidence_summary(scored_results)
            }
            
            logger.info(f"Document processed successfully in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {image_path}: {e}")
            raise
    
    def _extract_text(self, image_path: str) -> tuple[str, float]:
        """Extract text from image using OCR."""
        try:
            if hasattr(self.ocr, 'extract_text_with_coordinates'):
                # Use coordinate-based extraction for better confidence scoring
                ocr_result = self.ocr.extract_text_with_coordinates(image_path)
                extracted_text = ' '.join([item['text'] for item in ocr_result])
                ocr_confidence = np.mean([item['confidence'] for item in ocr_result]) if ocr_result else 0.0
            else:
                # Fallback to simple text extraction
                extracted_text = self.ocr.extract_text(image_path)
                ocr_confidence = 0.8  # Default confidence for simple extraction
            
            logger.info(f"Extracted {len(extracted_text)} characters with OCR confidence: {ocr_confidence:.3f}")
            return extracted_text, ocr_confidence
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    def _classify_document(self, text: str) -> tuple[str, float]:
        """Classify document type."""
        try:
            doc_type = self.classifier.predict(text)
            confidence = self.classifier.get_confidence_score(text)
            
            logger.info(f"Document classified as: {doc_type} (confidence: {confidence:.3f})")
            return doc_type, confidence
            
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            # Return default classification
            return "Unknown", 0.0
    
    def _create_extraction_results(
        self, 
        tax_entities: Dict[str, List[Dict]], 
        ocr_confidence: float
    ) -> List[ExtractionResult]:
        """Create ExtractionResult objects from NER entities."""
        results = []
        
        for field_name, entities in tax_entities.items():
            for entity in entities:
                result = ExtractionResult(
                    field_name=field_name,
                    value=entity['text'],
                    confidence=0.0,  # Will be calculated by confidence scorer
                    source_confidence=ocr_confidence,
                    model_confidence=entity['confidence'],
                    validation_score=0.0,  # Will be calculated by confidence scorer
                    start_position=entity.get('start'),
                    end_position=entity.get('end'),
                    metadata={'entity_data': entity}
                )
                results.append(result)
        
        return results
    
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            image_paths: List of paths to document images
            
        Returns:
            List of processing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing document {i+1}/{len(image_paths)}: {image_path}")
                result = self.process_document(image_path)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                # Add error result
                results.append({
                    'error': str(e),
                    'image_path': image_path,
                    'processing_metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'status': 'failed'
                    }
                })
        
        return results
    
    def get_structured_data(self, image_path: str) -> Dict[str, Union[str, None]]:
        """
        Get structured data in simplified format.
        
        Args:
            image_path: Path to the document image
            
        Returns:
            Dictionary with field names and their best values
        """
        result = self.process_document(image_path)
        
        structured_data = {}
        for field in result['extracted_fields']:
            field_name = field['field_name']
            if field['confidence'] > 0.5:  # Only include high-confidence fields
                structured_data[field_name] = field['value']
            else:
                structured_data[field_name] = None
        
        return structured_data
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing pipeline."""
        return {
            'classifier_info': self.classifier.get_model_info(),
            'ocr_processor': type(self.ocr).__name__,
            'confidence_weights': self.confidence_scorer.weights,
            'ner_model_path': self.ner_model_path,
            'classifier_model_path': self.classifier_model_path
        }
    
    def validate_document(self, image_path: str, expected_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate document processing results.
        
        Args:
            image_path: Path to the document image
            expected_type: Expected document type (optional)
            
        Returns:
            Validation results
        """
        result = self.process_document(image_path)
        
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'confidence_score': result['confidence_summary']['average_confidence']
        }
        
        # Check document type
        if expected_type and result['document_type'] != expected_type:
            validation['warnings'].append(
                f"Expected {expected_type}, got {result['document_type']}"
            )
        
        # Check confidence scores
        if validation['confidence_score'] < 0.5:
            validation['issues'].append("Low overall confidence score")
        
        # Check for missing critical fields
        critical_fields = ['SSN', 'WAGE_AMOUNT', 'EMPLOYER_NAME']
        missing_fields = []
        for field in critical_fields:
            field_found = any(
                f['field_name'] == field and f['confidence'] > 0.5 
                for f in result['extracted_fields']
            )
            if not field_found:
                missing_fields.append(field)
        
        if missing_fields:
            validation['warnings'].append(f"Missing critical fields: {', '.join(missing_fields)}")
        
        # Check processing time
        processing_time = result['processing_metadata']['processing_time_seconds']
        if processing_time > 10:  # More than 10 seconds
            validation['warnings'].append(f"Slow processing time: {processing_time:.2f}s")
        
        validation['is_valid'] = len(validation['issues']) == 0
        
        return validation
