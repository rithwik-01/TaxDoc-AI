#!/usr/bin/env python3
"""
Training script for Tax Field NER Model.

This script fine-tunes LegalBERT for tax-specific named entity recognition.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from taxdoc_ai.ner.field_extractor import TAX_ENTITY_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxNERDataset(Dataset):
    """Dataset class for tax NER training."""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.label2id = {label: i for i, label in enumerate(self._get_all_labels())}
        self.id2label = {i: label for label, i in self.label2id.items()}
    
    def _get_all_labels(self) -> List[str]:
        """Get all possible labels including BIO tags."""
        labels = ['O']  # Outside
        for entity_type in TAX_ENTITY_TYPES:
            labels.extend([f'B-{entity_type}', f'I-{entity_type}'])
        return labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            is_split_into_words=True
        )
        
        # Align labels with tokens
        word_ids = encoding.word_ids()
        aligned_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Special tokens
            else:
                if word_id < len(labels):
                    aligned_labels.append(self.label2id.get(labels[word_id], 0))
                else:
                    aligned_labels.append(-100)  # Padding
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

def create_synthetic_ner_data(num_samples: int = 500) -> Tuple[List[str], List[List[str]]]:
    """Create synthetic NER training data."""
    logger.info("Creating synthetic NER training data...")
    
    # Sample texts with entity annotations
    sample_data = [
        {
            "text": ["Employee", "Name", ":", "John", "Doe", "SSN", ":", "123-45-6789"],
            "labels": ["O", "O", "O", "B-EMPLOYEE_NAME", "I-EMPLOYEE_NAME", "O", "O", "B-SSN", "I-SSN", "I-SSN"]
        },
        {
            "text": ["Employer", ":", "ABC", "Corporation", "EIN", ":", "12-3456789"],
            "labels": ["O", "O", "B-EMPLOYER_NAME", "I-EMPLOYER_NAME", "O", "O", "B-EMPLOYER_EIN", "I-EMPLOYER_EIN"]
        },
        {
            "text": ["Wages", ":", "$", "50", ",", "000", ".", "00"],
            "labels": ["O", "O", "O", "B-WAGE_AMOUNT", "I-WAGE_AMOUNT", "I-WAGE_AMOUNT", "I-WAGE_AMOUNT", "I-WAGE_AMOUNT"]
        },
        {
            "text": ["Federal", "Tax", "Withheld", ":", "$", "5", ",", "000", ".", "00"],
            "labels": ["O", "O", "O", "O", "O", "B-FEDERAL_TAX", "I-FEDERAL_TAX", "I-FEDERAL_TAX", "I-FEDERAL_TAX", "I-FEDERAL_TAX"]
        },
        {
            "text": ["Date", ":", "12", "/", "31", "/", "2023"],
            "labels": ["O", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "I-DATE"]
        },
        {
            "text": ["Address", ":", "123", "Main", "St", ",", "Anytown", ",", "NY", "12345"],
            "labels": ["O", "O", "B-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS", "I-ADDRESS"]
        },
        {
            "text": ["Phone", ":", "555", "-", "123", "-", "4567"],
            "labels": ["O", "O", "B-PHONE_NUMBER", "I-PHONE_NUMBER", "I-PHONE_NUMBER", "I-PHONE_NUMBER", "I-PHONE_NUMBER"]
        },
        {
            "text": ["Email", ":", "john.doe", "@", "email.com"],
            "labels": ["O", "O", "B-EMAIL", "I-EMAIL", "I-EMAIL"]
        }
    ]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        # Select random sample
        sample = np.random.choice(sample_data)
        
        # Add variations
        variations = [
            sample["text"],
            [word.upper() for word in sample["text"]],
            [word.lower() for word in sample["text"]],
            sample["text"] + ["Additional", "text"],
            ["Document", ":"] + sample["text"]
        ]
        
        text_variation = np.random.choice(variations)
        label_variation = sample["labels"][:len(text_variation)]
        
        # Pad labels if needed
        while len(label_variation) < len(text_variation):
            label_variation.append("O")
        
        texts.append(text_variation)
        labels.append(label_variation)
    
    logger.info(f"Created {len(texts)} synthetic NER samples")
    return texts, labels

def load_ner_data(data_path: str) -> Tuple[List[str], List[List[str]]]:
    """Load NER training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    texts = data['texts']
    labels = data['labels']
    
    # Validate data
    assert len(texts) == len(labels), "Texts and labels must have same length"
    
    logger.info(f"Loaded {len(texts)} NER training samples")
    return texts, labels

def compute_metrics(eval_pred):
    """Compute metrics for NER evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Flatten
    true_predictions = [p for pred in true_predictions for p in pred]
    true_labels = [l for label in true_labels for l in label]
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='weighted')
    accuracy = accuracy_score(true_labels, true_predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_ner_model(
    texts: List[str],
    labels: List[List[str]],
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    validation_split: float = 0.2
):
    """Train the tax NER model."""
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=validation_split, random_state=42
    )
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")
    
    # Initialize tokenizer
    model_name = "jhu-clsp/LegalBert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = TaxNERDataset(train_texts, train_labels, tokenizer)
    val_dataset = TaxNERDataset(val_texts, val_labels, tokenizer)
    
    # Get label mappings
    label2id = train_dataset.label2id
    id2label = train_dataset.id2label
    
    # Initialize model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
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
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=learning_rate,
        report_to=None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    logger.info("Starting NER training...")
    trainer.train()
    
    # Save model
    model_path = os.path.join(output_dir, "tax_ner_model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Save label mappings
    with open(os.path.join(model_path, "label_mappings.json"), 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    
    # Evaluate model
    logger.info("Evaluating NER model...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    with open(os.path.join(output_dir, "ner_evaluation_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"NER training completed. Model saved to {model_path}")
    logger.info(f"Validation F1: {eval_results['eval_f1']:.4f}")
    logger.info(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    return model, eval_results

def main():
    parser = argparse.ArgumentParser(description="Train Tax NER Model")
    parser.add_argument("--data_path", type=str, help="Path to NER training data JSON file")
    parser.add_argument("--output_dir", type=str, default="./models/ner", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of synthetic samples")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create training data
    if args.synthetic or not args.data_path:
        texts, labels = create_synthetic_ner_data(args.num_samples)
    else:
        texts, labels = load_ner_data(args.data_path)
    
    # Train model
    model, eval_results = train_ner_model(
        texts=texts,
        labels=labels,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info("NER training completed successfully!")

if __name__ == "__main__":
    main()
