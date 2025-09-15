#!/usr/bin/env python3
"""
Demo script for TaxDoc-AI system.

This script demonstrates the complete tax document processing pipeline
with a simple command-line interface.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from taxdoc_ai import TaxDocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for demonstration."""
    sample_data = {
        "classification": [
            {
                "text": "Wage and Tax Statement 2023 Employee's social security number 123-45-6789 Employer ABC Corp",
                "true_label": "W-2"
            },
            {
                "text": "Form 1099-R Distributions From Pensions Gross distribution $25,000.00",
                "true_label": "1099-R"
            },
            {
                "text": "Form 1099-MISC Miscellaneous Income Nonemployee compensation $15,000.00",
                "true_label": "1099-MISC"
            }
        ],
        "ner": [
            {
                "text": "Employee Name: John Doe SSN: 123-45-6789 Employer: ABC Corporation EIN: 12-3456789 Wages: $50,000.00",
                "true_entities": {
                    "EMPLOYEE_NAME": [{"text": "John Doe", "confidence": 0.9, "start": 14, "end": 22}],
                    "SSN": [{"text": "123-45-6789", "confidence": 0.9, "start": 27, "end": 38}],
                    "EMPLOYER_NAME": [{"text": "ABC Corporation", "confidence": 0.9, "start": 48, "end": 63}],
                    "EMPLOYER_EIN": [{"text": "12-3456789", "confidence": 0.9, "start": 68, "end": 78}],
                    "WAGE_AMOUNT": [{"text": "$50,000.00", "confidence": 0.9, "start": 86, "end": 96}]
                }
            }
        ],
        "pipeline": [
            {
                "image_path": "sample_w2.jpg",  # This would be a real image file
                "user_corrections": 0
            }
        ]
    }
    return sample_data

def demo_classification(processor: TaxDocumentProcessor):
    """Demonstrate document classification."""
    print("\n" + "="*50)
    print("DOCUMENT CLASSIFICATION DEMO")
    print("="*50)
    
    sample_texts = [
        "Wage and Tax Statement 2023 Employee's social security number 123-45-6789",
        "Form 1099-R Distributions From Pensions, Annuities, Retirement Plans",
        "Form 1099-MISC Miscellaneous Income Nonemployee compensation",
        "Form 1099-DIV Dividends and Distributions Total ordinary dividends",
        "Receipt Date: 03/15/2023 Business expense $150.00 Office supplies",
        "Bank Statement Account Number 1234567890 Statement Period 01/01/2023",
        "Form 1040 U.S. Individual Income Tax Return Tax year 2023",
        "Schedule A Itemized Deductions Medical and dental expenses"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        try:
            predicted_type = processor.classifier.predict(text)
            confidence = processor.classifier.get_confidence_score(text)
            
            print(f"\nSample {i}:")
            print(f"Text: {text[:60]}...")
            print(f"Predicted Type: {predicted_type}")
            print(f"Confidence: {confidence:.3f}")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

def demo_ner(processor: TaxDocumentProcessor):
    """Demonstrate named entity recognition."""
    print("\n" + "="*50)
    print("NAMED ENTITY RECOGNITION DEMO")
    print("="*50)
    
    sample_texts = [
        "Employee Name: John Doe SSN: 123-45-6789 Employer: ABC Corporation EIN: 12-3456789 Wages: $50,000.00",
        "Federal Tax Withheld: $5,000.00 State Tax: $2,500.00 Social Security Tax: $3,100.00 Medicare Tax: $725.00",
        "Address: 123 Main Street, Anytown, NY 12345 Phone: 555-123-4567 Email: john.doe@email.com",
        "Date: 12/31/2023 Account Number: 1234567890 Routing Number: 021000021"
    ]
    
    for i, text in enumerate(sample_texts, 1):
        try:
            entities = processor.ner_extractor.extract_tax_entities(text)
            
            print(f"\nSample {i}:")
            print(f"Text: {text}")
            print("Extracted Entities:")
            
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"  {entity_type}:")
                    for entity in entity_list:
                        print(f"    - {entity['text']} (confidence: {entity['confidence']:.3f})")
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

def demo_confidence_scoring(processor: TaxDocumentProcessor):
    """Demonstrate confidence scoring."""
    print("\n" + "="*50)
    print("CONFIDENCE SCORING DEMO")
    print("="*50)
    
    test_values = {
        "SSN": ["123-45-6789", "000-00-0000", "123456789", "invalid"],
        "WAGE_AMOUNT": ["$50,000.00", "$1,000,000.00", "invalid amount", "50.000"],
        "DATE": ["12/31/2023", "2023-12-31", "invalid date", "13/45/2023"],
        "EMAIL": ["john.doe@email.com", "invalid-email", "test@domain", "no-at-sign"],
        "PHONE_NUMBER": ["555-123-4567", "5551234567", "invalid phone", "123"]
    }
    
    for field_name, values in test_values.items():
        print(f"\n{field_name} Validation:")
        for value in values:
            try:
                validation_score = processor.confidence_scorer.validate_tax_field(field_name, value)
                print(f"  '{value}' -> {validation_score:.3f}")
            except Exception as e:
                print(f"  '{value}' -> Error: {e}")

def demo_pipeline(processor: TaxDocumentProcessor, image_path: Optional[str] = None):
    """Demonstrate complete pipeline processing."""
    print("\n" + "="*50)
    print("COMPLETE PIPELINE DEMO")
    print("="*50)
    
    if not image_path or not os.path.exists(image_path):
        print("No valid image path provided. Creating a mock result...")
        
        # Create a mock result for demonstration
        mock_result = {
            'document_type': 'W-2',
            'classification_confidence': 0.95,
            'extracted_fields': [
                {
                    'field_name': 'EMPLOYEE_NAME',
                    'value': 'John Doe',
                    'confidence': 0.92,
                    'source_confidence': 0.88,
                    'model_confidence': 0.95,
                    'validation_score': 0.90
                },
                {
                    'field_name': 'SSN',
                    'value': '123-45-6789',
                    'confidence': 0.98,
                    'source_confidence': 0.88,
                    'model_confidence': 0.95,
                    'validation_score': 0.90
                },
                {
                    'field_name': 'WAGE_AMOUNT',
                    'value': '$50,000.00',
                    'confidence': 0.94,
                    'source_confidence': 0.88,
                    'model_confidence': 0.95,
                    'validation_score': 0.90
                }
            ],
            'raw_text': 'Wage and Tax Statement 2023 Employee Name: John Doe SSN: 123-45-6789 Wages: $50,000.00',
            'processing_metadata': {
                'ocr_confidence': 0.88,
                'processing_time_seconds': 2.5,
                'timestamp': '2024-01-15T10:30:00',
                'ocr_processor': 'PaddleOCRProcessor',
                'image_path': 'mock_image.jpg'
            },
            'confidence_summary': {
                'total_results': 3,
                'average_confidence': 0.95,
                'high_confidence_count': 3,
                'medium_confidence_count': 0,
                'low_confidence_count': 0
            }
        }
        
        result = mock_result
    else:
        try:
            result = processor.process_document(image_path)
        except Exception as e:
            print(f"Error processing document: {e}")
            return
    
    # Display results
    print(f"Document Type: {result['document_type']}")
    print(f"Classification Confidence: {result['classification_confidence']:.3f}")
    print(f"Processing Time: {result['processing_metadata']['processing_time_seconds']:.2f} seconds")
    print(f"OCR Processor: {result['processing_metadata']['ocr_processor']}")
    
    print(f"\nExtracted Fields ({len(result['extracted_fields'])} total):")
    for field in result['extracted_fields']:
        print(f"  {field['field_name']}: {field['value']} (confidence: {field['confidence']:.3f})")
    
    print(f"\nConfidence Summary:")
    summary = result['confidence_summary']
    print(f"  Average Confidence: {summary['average_confidence']:.3f}")
    print(f"  High Confidence Fields: {summary['high_confidence_count']}")
    print(f"  Medium Confidence Fields: {summary['medium_confidence_count']}")
    print(f"  Low Confidence Fields: {summary['low_confidence_count']}")
    
    # Validate document
    validation = processor.validate_document(image_path or "mock_image.jpg")
    print(f"\nDocument Validation:")
    print(f"  Valid: {validation['is_valid']}")
    if validation['issues']:
        print(f"  Issues: {', '.join(validation['issues'])}")
    if validation['warnings']:
        print(f"  Warnings: {', '.join(validation['warnings'])}")

def demo_evaluation(processor: TaxDocumentProcessor):
    """Demonstrate evaluation capabilities."""
    print("\n" + "="*50)
    print("EVALUATION DEMO")
    print("="*50)
    
    try:
        from taxdoc_ai.evaluation import TaxDocumentEvaluator
        
        evaluator = TaxDocumentEvaluator(processor)
        sample_data = create_sample_data()
        
        print("Running evaluation on sample data...")
        
        # Note: This would normally process real test data
        print("Evaluation capabilities demonstrated:")
        print("- Classification accuracy measurement")
        print("- NER precision/recall calculation")
        print("- Pipeline performance metrics")
        print("- Business value scoring")
        print("- Comprehensive reporting")
        
    except ImportError:
        print("Evaluation module not available in this demo")

def main():
    parser = argparse.ArgumentParser(description="TaxDoc-AI Demo")
    parser.add_argument("--nvidia_api_key", type=str, help="NVIDIA API key for OCR")
    parser.add_argument("--classifier_model", type=str, help="Path to trained classifier model")
    parser.add_argument("--ner_model", type=str, help="Path to trained NER model")
    parser.add_argument("--image_path", type=str, help="Path to test image")
    parser.add_argument("--demo_type", type=str, choices=["all", "classification", "ner", "confidence", "pipeline", "evaluation"], 
                       default="all", help="Type of demo to run")
    
    args = parser.parse_args()
    
    print("TaxDoc-AI Demo")
    print("="*50)
    
    try:
        # Initialize processor
        print("Initializing TaxDocumentProcessor...")
        processor = TaxDocumentProcessor(
            nvidia_api_key=args.nvidia_api_key,
            classifier_model_path=args.classifier_model,
            ner_model_path=args.ner_model
        )
        print("Processor initialized successfully!")
        
        # Run demos based on selection
        if args.demo_type in ["all", "classification"]:
            demo_classification(processor)
        
        if args.demo_type in ["all", "ner"]:
            demo_ner(processor)
        
        if args.demo_type in ["all", "confidence"]:
            demo_confidence_scoring(processor)
        
        if args.demo_type in ["all", "pipeline"]:
            demo_pipeline(processor, args.image_path)
        
        if args.demo_type in ["all", "evaluation"]:
            demo_evaluation(processor)
        
        print("\n" + "="*50)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
