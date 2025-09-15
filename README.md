# TaxDoc-AI: Intelligent Tax Document Classification & Extraction System

A comprehensive AI pipeline that automatically classifies tax documents and extracts structured data from various document types (W-2s, 1099s, receipts, bank statements) using state-of-the-art OCR and legal-domain NLP models.

## 🚀 Features

- **Document Classification**: Automatically identifies document types (W-2, 1099-R, 1099-MISC, etc.) using fine-tuned LegalBERT
- **Advanced OCR**: Supports both NVIDIA NIM API and PaddleOCR with intelligent fallback
- **Named Entity Recognition**: Extracts tax-specific fields (SSN, amounts, dates, addresses) with high accuracy
- **Confidence Scoring**: Multi-layered confidence system combining OCR, NER, and business logic validation
- **End-to-End Pipeline**: Complete processing from image to structured data
- **Comprehensive Evaluation**: Detailed metrics and performance analysis
- **Easy Integration**: Simple API and command-line interface

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- NVIDIA API key (optional, for NVIDIA OCR)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/taxdoc-ai.git
cd taxdoc-ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install the package:**
```bash
pip install -e .
```

## 🚀 Quick Start

### Basic Usage

```python
from taxdoc_ai import TaxDocumentProcessor

# Initialize processor
processor = TaxDocumentProcessor()

# Process a document
result = processor.process_document("path/to/document.jpg")

# Get structured data
structured_data = processor.get_structured_data("path/to/document.jpg")
print(structured_data)
```

### Command Line Interface

```bash
# Process a single document
taxdoc-ai process document.jpg --output results.json

# Process multiple documents
taxdoc-ai batch --input_dir ./documents --output batch_results.json

# Train models with synthetic data
taxdoc-ai train --train_classifier --train_ner --synthetic --num_samples 1000
```

### Demo Script

```bash
# Run comprehensive demo
python examples/demo.py

# Run specific demo components
python examples/demo.py --demo_type classification
python examples/demo.py --demo_type ner
python examples/demo.py --demo_type pipeline
```

## 🏗️ Architecture

### Core Components

1. **Document Classifier** (`TaxDocumentClassifier`)
   - Fine-tuned LegalBERT for tax document classification
   - Supports 8 document types: W-2, 1099-R, 1099-MISC, 1099-DIV, Receipt, Bank Statement, 1040, Schedule A

2. **OCR Processors** (`NVIDIAOCRProcessor`, `PaddleOCRProcessor`)
   - NVIDIA NIM API for high-accuracy OCR
   - PaddleOCR as open-source fallback
   - Automatic processor selection based on API key availability

3. **NER Extractor** (`TaxFieldExtractor`)
   - Fine-tuned LegalBERT for tax-specific entity recognition
   - Extracts 16 entity types: SSN, EIN, amounts, dates, addresses, etc.
   - Post-processing rules for improved accuracy

4. **Confidence Scorer** (`ConfidenceScorer`)
   - Multi-component confidence scoring
   - Business logic validation for tax fields
   - Weighted composite confidence calculation

5. **Main Pipeline** (`TaxDocumentProcessor`)
   - Orchestrates all components
   - End-to-end document processing
   - Batch processing capabilities

### Data Flow

```
Document Image → OCR → Text Extraction → Document Classification
                                                      ↓
Structured Data ← Confidence Scoring ← NER ← Classified Text
```

## 📊 Performance Metrics

### Expected Performance (with trained models)

- **Document Classification**: 95%+ accuracy
- **Field Extraction**: 85%+ accuracy with confidence > 0.8
- **Processing Time**: < 3 seconds per document
- **Scalability**: 1000+ documents per hour

### Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-score
- **NER**: Entity-level precision/recall, Exact match accuracy
- **Pipeline**: Processing time, Business value score, User correction rate

## 🎯 Supported Document Types

| Document Type | Description | Key Fields |
|---------------|-------------|------------|
| W-2 | Wage and Tax Statement | SSN, Employer Name, Wages, Tax Withheld |
| 1099-R | Retirement Distributions | Gross Distribution, Taxable Amount |
| 1099-MISC | Miscellaneous Income | Nonemployee Compensation, Rents |
| 1099-DIV | Dividends and Distributions | Ordinary Dividends, Qualified Dividends |
| Receipt | Business Expense Receipts | Date, Amount, Vendor |
| Bank Statement | Account Statements | Account Number, Transactions |
| 1040 | Individual Tax Return | AGI, Taxable Income |
| Schedule A | Itemized Deductions | Medical Expenses, State Taxes |

## 🔧 Configuration

### Environment Variables

```bash
# NVIDIA API Configuration
export NVIDIA_API_KEY="your_nvidia_api_key"

# Model Paths
export CLASSIFIER_MODEL_PATH="./models/classifier/tax_classifier.pth"
export NER_MODEL_PATH="./models/ner/tax_ner_model"
```

### Configuration File

Create `config.json`:

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

## 🎓 Training Models

### Train Document Classifier

```bash
python scripts/train_classifier.py \
    --synthetic \
    --num_samples 1000 \
    --num_epochs 3 \
    --output_dir ./models/classifier
```

### Train NER Model

```bash
python scripts/train_ner.py \
    --synthetic \
    --num_samples 500 \
    --num_epochs 3 \
    --output_dir ./models/ner
```

### Custom Training Data

Prepare your training data in JSON format:

**Classification Data:**
```json
{
  "texts": ["Document text 1", "Document text 2", ...],
  "labels": [0, 1, 2, ...],
  "doc_types": ["W-2", "1099-R", "1099-MISC", ...]
}
```

**NER Data:**
```json
{
  "texts": [["Token", "1", "Token", "2", ...], ...],
  "labels": [["O", "B-SSN", "I-SSN", "O", ...], ...]
}
```

## 📈 Evaluation

### Run Comprehensive Evaluation

```python
from taxdoc_ai.evaluation import TaxDocumentEvaluator

evaluator = TaxDocumentEvaluator(processor)
results = evaluator.evaluate_all(test_data)
```

### Generate Evaluation Report

```bash
python -c "
from taxdoc_ai.evaluation import TaxDocumentEvaluator
from taxdoc_ai import TaxDocumentProcessor

processor = TaxDocumentProcessor()
evaluator = TaxDocumentEvaluator(processor)
# Load your test data and run evaluation
"
```

## 🔍 API Reference

### TaxDocumentProcessor

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
    
    def process_document(self, image_path: str) -> Dict[str, Any]
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]
    def get_structured_data(self, image_path: str) -> Dict[str, Union[str, None]]
    def validate_document(self, image_path: str, expected_type: Optional[str] = None) -> Dict[str, Any]
```

### ConfidenceScorer

```python
class ConfidenceScorer:
    def __init__(self, weights: Optional[Dict[str, float]] = None)
    def calculate_composite_confidence(self, result: ExtractionResult) -> float
    def validate_tax_field(self, field_name: str, value: str) -> float
    def score_extraction_batch(self, results: List[ExtractionResult]) -> List[ExtractionResult]
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_classifier.py
pytest tests/test_ocr.py
pytest tests/test_ner.py
pytest tests/test_pipeline.py
```

### Test Coverage

```bash
pytest --cov=taxdoc_ai tests/
```

## 📝 Examples

### Example 1: Basic Document Processing

```python
from taxdoc_ai import TaxDocumentProcessor

# Initialize processor
processor = TaxDocumentProcessor()

# Process a W-2 form
result = processor.process_document("w2_form.jpg")

print(f"Document Type: {result['document_type']}")
print(f"Classification Confidence: {result['classification_confidence']:.3f}")

for field in result['extracted_fields']:
    if field['confidence'] > 0.8:
        print(f"{field['field_name']}: {field['value']}")
```

### Example 2: Batch Processing

```python
import glob
from taxdoc_ai import TaxDocumentProcessor

processor = TaxDocumentProcessor()

# Process all images in a directory
image_paths = glob.glob("documents/*.jpg")
results = processor.process_batch(image_paths)

# Analyze results
for i, result in enumerate(results):
    print(f"Document {i+1}: {result['document_type']} "
          f"(confidence: {result['classification_confidence']:.3f})")
```

### Example 3: Custom Confidence Weights

```python
from taxdoc_ai import TaxDocumentProcessor

# Custom confidence weights
confidence_weights = {
    'ocr': 0.2,
    'ner': 0.5,
    'validation': 0.3
}

processor = TaxDocumentProcessor(confidence_weights=confidence_weights)
result = processor.process_document("document.jpg")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
black src/
flake8 src/
mypy src/

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LegalBERT](https://github.com/reglab/LegalBERT) for the base model
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for open-source OCR
- [NVIDIA NIM](https://www.nvidia.com/en-us/ai-data-science/nim/) for high-accuracy OCR API
- [Transformers](https://huggingface.co/transformers/) for the NLP framework

---

**TaxDoc-AI** - Making tax document processing intelligent, accurate, and efficient.
