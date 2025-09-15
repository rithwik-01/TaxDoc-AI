# TaxDoc-AI API Reference

This document provides comprehensive API reference for the TaxDoc-AI system.

## Core Classes

### TaxDocumentProcessor

The main pipeline class that orchestrates all components for tax document processing.

```python
class TaxDocumentProcessor:
    def __init__(
        self,
        nvidia_api_key: Optional[str] = None,
        classifier_model_path: Optional[str] = None,
        ner_model_path: Optional[str] = None,
        ocr_processor_type: str = "auto",
        confidence_weights: Optional[Dict[str, float]] = None
    )
```

**Parameters:**
- `nvidia_api_key` (Optional[str]): NVIDIA API key for OCR processing
- `classifier_model_path` (Optional[str]): Path to trained classifier model
- `ner_model_path` (Optional[str]): Path to trained NER model
- `ocr_processor_type` (str): Type of OCR processor ("nvidia", "paddle", "auto")
- `confidence_weights` (Optional[Dict[str, float]]): Weights for confidence scoring

**Methods:**

#### process_document(image_path: str) -> Dict[str, Any]

Process a single tax document and return comprehensive results.

**Parameters:**
- `image_path` (str): Path to the document image

**Returns:**
- `Dict[str, Any]`: Processing results including document type, extracted fields, confidence scores, and metadata

**Example:**
```python
processor = TaxDocumentProcessor()
result = processor.process_document("w2_form.jpg")

print(f"Document Type: {result['document_type']}")
print(f"Confidence: {result['classification_confidence']:.3f}")
```

#### process_batch(image_paths: List[str]) -> List[Dict[str, Any]]

Process multiple documents in batch.

**Parameters:**
- `image_paths` (List[str]): List of paths to document images

**Returns:**
- `List[Dict[str, Any]]`: List of processing results for each document

#### get_structured_data(image_path: str) -> Dict[str, Union[str, None]]

Get structured data in simplified format.

**Parameters:**
- `image_path` (str): Path to the document image

**Returns:**
- `Dict[str, Union[str, None]]`: Dictionary with field names and their best values

#### validate_document(image_path: str, expected_type: Optional[str] = None) -> Dict[str, Any]

Validate document processing results.

**Parameters:**
- `image_path` (str): Path to the document image
- `expected_type` (Optional[str]): Expected document type

**Returns:**
- `Dict[str, Any]`: Validation results with issues and warnings

### TaxDocumentClassifier

Neural network classifier for tax document types using LegalBERT.

```python
class TaxDocumentClassifier(nn.Module):
    def __init__(
        self, 
        num_classes: int = 8, 
        model_name: str = "jhu-clsp/LegalBert", 
        dropout_rate: float = 0.3
    )
```

**Methods:**

#### predict(text: str, return_probabilities: bool = False) -> Union[str, Dict[str, float]]

Predict document type for given text.

**Parameters:**
- `text` (str): Input text to classify
- `return_probabilities` (bool): Whether to return class probabilities

**Returns:**
- `Union[str, Dict[str, float]]`: Predicted document type or probability dictionary

#### predict_batch(texts: List[str], return_probabilities: bool = False) -> Union[List[str], List[Dict[str, float]]]

Predict document types for a batch of texts.

#### get_confidence_score(text: str) -> float

Get confidence score for the prediction.

### TaxFieldExtractor

Named Entity Recognition extractor for tax-specific fields.

```python
class TaxFieldExtractor:
    def __init__(self, model_path: Optional[str] = None, model_name: str = "jhu-clsp/LegalBert")
```

**Methods:**

#### extract_tax_entities(text: str) -> Dict[str, List[Dict]]

Extract tax-specific entities from text.

**Parameters:**
- `text` (str): Input text to process

**Returns:**
- `Dict[str, List[Dict]]`: Dictionary mapping entity types to lists of extracted entities

#### extract_structured_data(text: str) -> Dict[str, Union[str, float, None]]

Extract structured data in simplified format.

### ConfidenceScorer

Confidence scoring system for tax field extraction results.

```python
class ConfidenceScorer:
    def __init__(self, weights: Optional[Dict[str, float]] = None)
```

**Methods:**

#### calculate_composite_confidence(result: ExtractionResult) -> float

Calculate weighted composite confidence score.

#### validate_tax_field(field_name: str, value: str) -> float

Validate tax field using business logic.

#### score_extraction_batch(results: List[ExtractionResult]) -> List[ExtractionResult]

Score a batch of extraction results.

### OCR Processors

#### NVIDIAOCRProcessor

High-accuracy OCR processor using NVIDIA NIM API.

```python
class NVIDIAOCRProcessor(BaseOCRProcessor):
    def __init__(self, api_key: str, base_url: str = "https://ai.api.nvidia.com/v1/cv/nvidia/ocr")
```

#### PaddleOCRProcessor

Open-source OCR processor using PaddleOCR.

```python
class PaddleOCRProcessor(BaseOCRProcessor):
    def __init__(self, use_angle_cls: bool = True, lang: str = 'en')
```

## Data Classes

### ExtractionResult

Data class representing an extraction result with confidence scores.

```python
@dataclass
class ExtractionResult:
    field_name: str
    value: str
    confidence: float
    source_confidence: float  # OCR confidence
    model_confidence: float   # NER confidence
    validation_score: float   # Business logic validation
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    metadata: Optional[Dict] = None
```

## Evaluation Classes

### TaxDocumentEvaluator

Comprehensive evaluator for the tax document processing system.

```python
class TaxDocumentEvaluator:
    def __init__(self, processor: TaxDocumentProcessor)
```

**Methods:**

#### evaluate_classification(test_data: List[Dict[str, Any]], save_results: bool = True, output_dir: str = "./evaluation_results") -> Dict[str, Any]

Evaluate document classification performance.

#### evaluate_ner(test_data: List[Dict[str, Any]], save_results: bool = True, output_dir: str = "./evaluation_results") -> Dict[str, Any]

Evaluate named entity recognition performance.

#### evaluate_pipeline(test_data: List[Dict[str, Any]], save_results: bool = True, output_dir: str = "./evaluation_results") -> Dict[str, Any]

Evaluate end-to-end pipeline performance.

#### evaluate_all(test_data: Dict[str, List[Dict[str, Any]]], save_results: bool = True, output_dir: str = "./evaluation_results") -> Dict[str, Any]

Run comprehensive evaluation on all components.

## Constants

### Document Types

```python
DOCUMENT_TYPES = [
    'W-2',
    '1099-R', 
    '1099-MISC',
    '1099-DIV',
    'Receipt',
    'Bank Statement',
    '1040',
    'Schedule A'
]
```

### Entity Types

```python
TAX_ENTITY_TYPES = [
    'SSN',
    'EMPLOYER_NAME', 
    'WAGE_AMOUNT',
    'TAX_WITHHELD',
    'ADDRESS',
    'DATE',
    'EMPLOYER_EIN',
    'EMPLOYEE_NAME',
    'FEDERAL_TAX',
    'STATE_TAX',
    'SOCIAL_SECURITY_TAX',
    'MEDICARE_TAX',
    'ACCOUNT_NUMBER',
    'ROUTING_NUMBER',
    'PHONE_NUMBER',
    'EMAIL'
]
```

## Error Handling

### Common Exceptions

- `FileNotFoundError`: When image file doesn't exist
- `ValueError`: When invalid parameters are provided
- `ImportError`: When required dependencies are missing
- `RuntimeError`: When model loading fails

### Error Handling Example

```python
try:
    processor = TaxDocumentProcessor()
    result = processor.process_document("document.jpg")
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Memory Usage

- LegalBERT model: ~1.5GB RAM
- PaddleOCR: ~500MB RAM
- Total system: ~2-3GB RAM

### Processing Time

- Document classification: ~0.5 seconds
- OCR processing: ~1-2 seconds
- NER extraction: ~0.5 seconds
- Total pipeline: ~2-3 seconds per document

### Optimization Tips

1. Use GPU acceleration when available
2. Batch process multiple documents
3. Cache model loading for repeated use
4. Use NVIDIA OCR for better accuracy when API key is available

## Configuration

### Environment Variables

```bash
export NVIDIA_API_KEY="your_nvidia_api_key"
export CLASSIFIER_MODEL_PATH="./models/classifier/tax_classifier.pth"
export NER_MODEL_PATH="./models/ner/tax_ner_model"
```

### Configuration File

```json
{
  "ocr": {
    "processor_type": "auto",
    "nvidia_api_key": "your_api_key_here"
  },
  "models": {
    "classifier_path": "./models/classifier/tax_classifier.pth",
    "ner_path": "./models/ner/tax_ner_model"
  },
  "confidence": {
    "weights": {
      "ocr": 0.3,
      "ner": 0.4,
      "validation": 0.3
    }
  }
}
```
