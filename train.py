import os
import re
import torch
import pandas as pd
from classifier import Transformer
from sklearn.metrics import classification_report

def replace_urls(text, replace_with="<URL>"):
    """Replace all URLs in text with a placeholder."""
    URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
    return URL_REGEX.sub(replace_with, text)
    
def preprocess_text(text):  
    """Clean text by removing extra spaces, replacing mentions, and masking URLs."""
    text = text.strip()
    text = re.sub(r'@[a-zA-Z0-9_]+', 'USER', text)  # Replace mentions with 'USER'
    return replace_urls(text, replace_with="URL")

def load_data(filepath):
    """Load and preprocess dataset."""
    df = pd.read_csv(filepath)
    df = df.rename(columns={"tweet": "text", "label": "labels"})
    df["text"] = df["text"].apply(preprocess_text)
    return df

# Load datasets
train_data = load_data("text_dataset/train.csv")
test_data = load_data("text_dataset/test.csv")

# Initialize and train model
model = Transformer(train_data, test_data, "FacebookAI/roberta-base")
model.train()

# Evaluate model
class_report = model.predict()
print(class_report)

# Save the trained model
torch.save(model, "roberta_tweet_classification.pth")
print("Model saved to roberta_tweet_classification.pth")
