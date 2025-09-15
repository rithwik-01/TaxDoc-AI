#!/usr/bin/env python3
"""
Training script for Tax Document Classifier.

This script fine-tunes LegalBERT for tax document classification.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from taxdoc_ai.classifiers.document_classifier import TaxDocumentClassifier, DOCUMENT_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxDocumentDataset(Dataset):
    """Dataset class for tax document classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_training_data(data_path: str) -> Tuple[List[str], List[int]]:
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    texts = data['texts']
    labels = data['labels']
    
    # Validate data
    assert len(texts) == len(labels), "Texts and labels must have same length"
    assert all(isinstance(label, int) for label in labels), "Labels must be integers"
    assert all(0 <= label < len(DOCUMENT_TYPES) for label in labels), f"Labels must be 0-{len(DOCUMENT_TYPES)-1}"
    
    logger.info(f"Loaded {len(texts)} training samples")
    return texts, labels

def create_synthetic_data(num_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """Create synthetic training data for demonstration."""
    logger.info("Creating synthetic training data...")
    
    # Sample texts for different document types
    sample_texts = {
        0: [  # W-2
            "Wage and Tax Statement 2023 Employee's social security number 123-45-6789",
            "Form W-2 Wage and Tax Statement Employer identification number 12-3456789",
            "Wages, tips, other compensation $50,000.00 Federal income tax withheld $5,000.00"
        ],
        1: [  # 1099-R
            "Form 1099-R Distributions From Pensions, Annuities, Retirement or Profit-Sharing Plans",
            "Gross distribution $25,000.00 Taxable amount $20,000.00",
            "Pension distribution 1099-R retirement plan distribution"
        ],
        2: [  # 1099-MISC
            "Form 1099-MISC Miscellaneous Income Nonemployee compensation $15,000.00",
            "Rents $12,000.00 Royalties $3,000.00 Other income $2,000.00",
            "1099-MISC miscellaneous income contractor payments"
        ],
        3: [  # 1099-DIV
            "Form 1099-DIV Dividends and Distributions Total ordinary dividends $1,500.00",
            "Qualified dividends $1,200.00 Capital gain distributions $300.00",
            "1099-DIV dividend income investment distributions"
        ],
        4: [  # Receipt
            "Receipt Date: 03/15/2023 Business expense $150.00 Office supplies",
            "Restaurant receipt $45.67 Business meal expense",
            "Travel expense receipt hotel accommodation $200.00"
        ],
        5: [  # Bank Statement
            "Bank Statement Account Number 1234567890 Statement Period 01/01/2023 - 01/31/2023",
            "Beginning Balance $5,000.00 Ending Balance $4,500.00",
            "Monthly bank statement checking account transactions"
        ],
        6: [  # 1040
            "Form 1040 U.S. Individual Income Tax Return Tax year 2023",
            "Adjusted gross income $75,000.00 Taxable income $60,000.00",
            "1040 individual tax return federal income tax"
        ],
        7: [  # Schedule A
            "Schedule A Itemized Deductions Medical and dental expenses $2,000.00",
            "State and local taxes $8,000.00 Mortgage interest $12,000.00",
            "Schedule A itemized deductions charitable contributions"
        ]
    }
    
    texts = []
    labels = []
    
    for label, examples in sample_texts.items():
        # Generate variations of each example
        for _ in range(num_samples // len(DOCUMENT_TYPES)):
            base_text = np.random.choice(examples)
            
            # Add some variation
            variations = [
                base_text,
                base_text.upper(),
                base_text.lower(),
                base_text + " Additional information here.",
                "Document: " + base_text,
                base_text.replace("2023", "2024"),
                base_text.replace("$", "USD ")
            ]
            
            texts.append(np.random.choice(variations))
            labels.append(label)
    
    logger.info(f"Created {len(texts)} synthetic samples")
    return texts, labels

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
    }

def train_classifier(
    texts: List[str],
    labels: List[int],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    validation_split: float = 0.2
):
    """Train the tax document classifier."""
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=validation_split, random_state=42, stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Initialize model and tokenizer
    model_name = "jhu-clsp/LegalBert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = TaxDocumentDataset(train_texts, train_labels, tokenizer)
    val_dataset = TaxDocumentDataset(val_texts, val_labels, tokenizer)
    
    # Initialize model
    model = TaxDocumentClassifier(num_classes=len(DOCUMENT_TYPES))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        learning_rate=learning_rate,
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    model_path = os.path.join(output_dir, "tax_classifier.pth")
    model.save_model(model_path)
    
    # Evaluate on validation set
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Generate detailed classification report
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    report = classification_report(val_labels, y_pred, target_names=DOCUMENT_TYPES)
    logger.info(f"Classification Report:\n{report}")
    
    # Save evaluation results
    eval_results['classification_report'] = report
    eval_results['document_types'] = DOCUMENT_TYPES
    
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Training completed. Model saved to {model_path}")
    logger.info(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")
    
    return model, eval_results

def main():
    parser = argparse.ArgumentParser(description="Train Tax Document Classifier")
    parser.add_argument("--data_path", type=str, help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, default="./models/classifier", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of synthetic samples")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create training data
    if args.synthetic or not args.data_path:
        texts, labels = create_synthetic_data(args.num_samples)
    else:
        texts, labels = load_training_data(args.data_path)
    
    # Train model
    model, eval_results = train_classifier(
        texts=texts,
        labels=labels,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
