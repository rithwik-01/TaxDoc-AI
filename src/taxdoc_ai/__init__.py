"""
TaxDoc-AI: Intelligent Tax Document Classification & Extraction System

A comprehensive AI pipeline that automatically classifies tax documents and extracts 
structured data from various document types using state-of-the-art OCR and legal-domain NLP models.
"""

__version__ = "1.0.0"
__author__ = "TaxDoc-AI Team"

from .pipeline.tax_document_processor import TaxDocumentProcessor
from .classifiers.document_classifier import TaxDocumentClassifier
from .ocr.ocr_processors import NVIDIAOCRProcessor, PaddleOCRProcessor
from .ner.field_extractor import TaxFieldExtractor
from .scoring.confidence_scorer import ConfidenceScorer, ExtractionResult

__all__ = [
    "TaxDocumentProcessor",
    "TaxDocumentClassifier", 
    "NVIDIAOCRProcessor",
    "PaddleOCRProcessor",
    "TaxFieldExtractor",
    "ConfidenceScorer",
    "ExtractionResult"
]
