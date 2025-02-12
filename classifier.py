import re
import torch
import evaluate
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_scheduler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Transformer:
    def __init__(self, training_data, testing_data, model_name):
        self.label2id = {'human': 0, 'ai': 1}
        self.id2label = {0: 'human', 1: 'ai'}

        # Convert labels
        self.training_data = training_data.copy()
        self.testing_data = testing_data.copy()
        self.training_data["labels"] = self.training_data["labels"].map(self.label2id)
        self.testing_data["labels"] = self.testing_data["labels"].map(self.label2id)

        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.best_f1 = None
        self.best_model = None
        self.best_epoch = None

    def preprocess(self, examples):
        """Tokenize input text."""
        return self.tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)

    def train(self):
        """Train the Transformer model."""
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(
            self.model_name, id2label=self.id2label, num_labels=len(self.label2id), label2id=self.label2id
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)

        # Split training data into train/validation sets
        self.training_data, val_data = train_test_split(
            self.training_data, test_size=0.2, stratify=self.training_data["labels"], random_state=256
        )

        # Convert to Hugging Face dataset format
        train_dataset = Dataset.from_pandas(self.training_data)
        val_dataset = Dataset.from_pandas(val_data)

        # Tokenize datasets
        train_dataset = train_dataset.map(self.preprocess, batched=True)
        val_dataset = val_dataset.map(self.preprocess, batched=True)

        # Remove unnecessary columns
        train_dataset = train_dataset.remove_columns(["text", "__index_level_0__"])
        val_dataset = val_dataset.remove_columns(["text", "__index_level_0__"])

        # Convert to PyTorch tensors
        train_dataset.set_format("torch")
        val_dataset.set_format("torch")

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Optimizer and Scheduler
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        num_epochs = 3
        num_training_steps = num_epochs * len(train_loader)
        lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.model.to(device)
        progress_bar = tqdm(range(num_training_steps))

        # Training Loop
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                if isinstance(batch, list):
                    batch = batch[0]
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluate model on validation set
            f1 = self.evaluate(val_loader)['f1']

            # Save the best model
            if self.best_f1 is None or self.best_f1 < f1:
                self.best_model = self.model
                self.best_f1 = f1
                self.best_epoch = epoch

        print(f"Best model F1-score: {self.best_f1} at epoch {self.best_epoch}")

    def evaluate(self, dataloader):
        """Evaluate the model using F1-score."""
        self.model.eval()
        metric = evaluate.load("f1")

        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        return metric.compute(average="macro")

    def predict(self):
        """Predict on the test set and return classification report."""
        self.model = self.best_model or self.model  # Use best model if available

        # Prepare test dataset
        test_dataset = Dataset.from_pandas(self.testing_data)
        test_dataset = test_dataset.map(self.preprocess, batched=True)
        test_dataset = test_dataset.remove_columns(["text"])
        test_dataset.set_format("torch")
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        self.model.eval()
        predicted_labels = []
        gold_labels = []

        for batch in tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            predictions = torch.argmax(outputs.logits, dim=-1)
            predicted_labels += predictions.tolist()
            gold_labels += batch["labels"].tolist()

        return classification_report(gold_labels, predicted_labels, digits=3)
