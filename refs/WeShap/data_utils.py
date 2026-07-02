import re
import numpy as np
from wrench.dataset import load_dataset, BaseDataset
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import pdb
import torch
from pathlib import Path
import os
import json
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

def preprocess_image_dataset(dataset, image_root):
    """
    Preprocess an image dataset in wrench format, extract raw features from image paths and store them in numpy array
    """
    # Define the image transformations (as required by ResNet-50)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_image(image_path):
        """Load an image, preprocess it, and extract features."""
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add a batch dimension
        return image

    image_list = []
    for i in range(len(dataset)):
        image_name = dataset.examples[i]["image"]
        image_path = Path(image_root) / image_name
        image = preprocess_image(image_path)
        image_list.append(image)

    image_feature = np.concatenate(image_list, axis=0)
    return image_feature

class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (numpy array or torch tensor): The input features.
            labels (numpy array or torch tensor): The labels.
        """
        self.features = torch.tensor(features, dtype=torch.float32) if isinstance(features, np.ndarray) else features
        self.labels = torch.tensor(labels, dtype=torch.long) if isinstance(labels, np.ndarray) else labels

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.features)

    def __getitem__(self, idx):
        """Generates one sample of data."""
        return self.features[idx], self.labels[idx]


def extract_features(sentences, tokenizer, model, batch_size=32):
    # Tokenize all sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Initialize a list to store our sentence embeddings
    all_sentence_embeddings = []

    for i in range(0, len(sentences), batch_size):
        # Prepare batch inputs
        batch_input = {k: v[i:i + batch_size] for k, v in encoded_input.items()}

        # Move batch to the same device as disc_model
        batch_input = {k: v.to(model.device) for k, v in batch_input.items()}

        with torch.no_grad():
            # Get disc_model outputs for the current batch
            outputs = model(**batch_input, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Aggregate the hidden states for the batch, e.g., mean of the last layer's output
            batch_embeddings = torch.mean(hidden_states[-1], dim=1)

            all_sentence_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings to have a single matrix
    sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0).numpy()

    return sentence_embeddings


def load_wrench_data(data_root, dataset_name, feature, image_root=None):
    if feature == "bert":
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn="bert", cache_name="bert")
    elif feature == "sentence-bert":
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,extract_feature=False)
        cache_path = Path(data_root) / dataset_name / "sentence_bert_cache"
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        train_cache_path = cache_path / "train_features.npy"
        valid_cache_path = cache_path / "valid_features.npy"
        test_cache_path = cache_path / "test_features.npy"

        if os.path.exists(train_cache_path):
            train_dataset.features = np.load(train_cache_path, allow_pickle=True)
            valid_dataset.features = np.load(valid_cache_path, allow_pickle=True)
            test_dataset.features = np.load(test_cache_path, allow_pickle=True)
        else:
            embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
            train_sentences = [train_dataset.examples[i]["text"] for i in range(len(train_dataset))]
            valid_sentences = [valid_dataset.examples[i]["text"] for i in range(len(valid_dataset))]
            test_sentences = [test_dataset.examples[i]["text"] for i in range(len(test_dataset))]
            train_dataset.features = embedding_model.encode(train_sentences)
            valid_dataset.features = embedding_model.encode(valid_sentences)
            test_dataset.features = embedding_model.encode(test_sentences)
            np.save(train_cache_path, train_dataset.features)
            np.save(valid_cache_path, valid_dataset.features)
            np.save(test_cache_path, test_dataset.features)

    elif feature == "bertweet-sentiment":
        cache_path = Path(data_root) / dataset_name / "bertweet_sentiment_cache"
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,extract_feature=False)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        train_cache_path = cache_path / "train_features.npy"
        valid_cache_path = cache_path / "valid_features.npy"
        test_cache_path = cache_path / "test_features.npy"

        if os.path.exists(train_cache_path):
            train_dataset.features = np.load(train_cache_path, allow_pickle=True)
            valid_dataset.features = np.load(valid_cache_path, allow_pickle=True)
            test_dataset.features = np.load(test_cache_path, allow_pickle=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
            model = AutoModelForSequenceClassification.from_pretrained(
                "finiteautomata/bertweet-base-sentiment-analysis")
            train_sentences = [train_dataset.examples[i]["text"] for i in range(len(train_dataset))]
            valid_sentences = [valid_dataset.examples[i]["text"] for i in range(len(valid_dataset))]
            test_sentences = [test_dataset.examples[i]["text"] for i in range(len(test_dataset))]
            train_dataset.features = extract_features(train_sentences, tokenizer, model)
            valid_dataset.features = extract_features(valid_sentences, tokenizer, model)
            test_dataset.features = extract_features(test_sentences, tokenizer, model)
            np.save(train_cache_path, train_dataset.features)
            np.save(valid_cache_path, valid_dataset.features)
            np.save(test_cache_path, test_dataset.features)

    else:
        train_dataset, valid_dataset, test_dataset = load_dataset(data_root, dataset_name,
                                                                  extract_feature=True, extract_fn=feature)

    if dataset_name in ["indoor-outdoor", "voc07-animal"]:
        train_raw = preprocess_image_dataset(train_dataset, image_root)
        valid_raw = preprocess_image_dataset(valid_dataset, image_root)
        test_raw = preprocess_image_dataset(test_dataset, image_root)
        train_dataset.images = train_raw
        valid_dataset.images = valid_raw
        test_dataset.images = test_raw

    return train_dataset, valid_dataset, test_dataset