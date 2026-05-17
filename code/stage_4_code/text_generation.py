import os
import re
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# Throw and catch errors incase stopwords isn't already downloaded

from nltk.corpus import stopwords
from tqdm import tqdm


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# This is the directory for my (Ethan W.) computer to find the train and test data.
# May need to change to appropriate directory if you want to run it.
TRAIN_DIR = r"C:\Users\User\OneDrive\Desktop\School\ECS 170\ECS_170\Given Stuff\Project_Stage_4\stage_4_data\text_classification\train"
TEST_DIR = r"C:\Users\User\OneDrive\Desktop\School\ECS 170\ECS_170\Given Stuff\Project_Stage_4\stage_4_data\text_classification\test"


MAX_VOCAB = 20000
MAX_LEN = 250
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
EPOCHS = 10
LR = 0.001

MODEL_TYPE = "GRU"   # Change to RNN / LSTM / GRU
# Required to test all of these and mark their evaluation metrics.
# KEEP SAME AS text_classification.py


# Text Cleaning (filters out stop words and punctuations and also normalizes the words to fix misspelling
# or noisy tokens)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()

    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return words



# Load Dataset
def load_reviews(directory):

    texts = []
    labels = []

    for label_type in ["pos", "neg"]:

        folder = os.path.join(directory, label_type)

        for file in os.listdir(folder):

            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf8") as f:
                text = f.read()

            tokens = clean_text(text)

            texts.append(tokens)

            labels.append(1 if label_type == "pos" else 0)

    return texts, labels

print("Loading data...")

train_texts, train_labels = load_reviews(TRAIN_DIR)
test_texts, test_labels = load_reviews(TEST_DIR)



# Build Vocabulary
counter = Counter()

for text in train_texts:
    counter.update(text)

vocab = counter.most_common(MAX_VOCAB - 2)

word2idx = {
    "<PAD>": 0,
    "<UNK>": 1
}

for idx, (word, _) in enumerate(vocab, start=2):
    word2idx[word] = idx



# Encode Sequences
def encode(text):

    seq = [word2idx.get(w, 1) for w in text]

    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]

    return seq

X_train = [encode(x) for x in train_texts]
X_test = [encode(x) for x in test_texts]



# Dataset Class
class ReviewDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

train_dataset = ReviewDataset(X_train, train_labels)
test_dataset = ReviewDataset(X_test, test_labels)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE)


# Model
class TextClassifier(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 num_layers,
                 model_type="RNN"):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if model_type == "RNN":

            self.rnn = nn.RNN(embed_dim,
                              hidden_dim,
                              num_layers=num_layers,
                              batch_first=True)

        elif model_type == "LSTM":

            self.rnn = nn.LSTM(embed_dim,
                               hidden_dim,
                               num_layers=num_layers,
                               batch_first=True)

        elif model_type == "GRU":

            self.rnn = nn.GRU(embed_dim,
                              hidden_dim,
                              num_layers=num_layers,
                              batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()

        self.model_type = model_type

    def forward(self, x):

        x = self.embedding(x)

        out, hidden = self.rnn(x)

        if self.model_type == "LSTM":
            hidden = hidden[0]

        hidden = hidden[-1]

        out = self.fc(hidden)

        return self.sigmoid(out).squeeze()

model = TextClassifier(
    vocab_size=len(word2idx),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    model_type=MODEL_TYPE
).to(DEVICE)

criterion = nn.BCELoss()

# Adam optimizer, could change
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Training
loss_history = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X_batch, y_batch in tqdm(train_loader):

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")


# Plot Loss Curve
plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{MODEL_TYPE} Training Loss")

plt.savefig("outputs/classifier_loss.png")


# Evaluations
model.eval()

predictions = []
ground_truth = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        outputs = model(X_batch)

        preds = (outputs >= 0.5).int().cpu().numpy()

        predictions.extend(preds)

        ground_truth.extend(y_batch.numpy())


# Evaulation Metrics
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, average="macro")
recall = recall_score(ground_truth, predictions, average="macro")
f1 = f1_score(ground_truth, predictions, average="macro")

print("\nEvaluation Results")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

torch.save(model.state_dict(),
           f"models/{MODEL_TYPE.lower()}_classifier.pth")
