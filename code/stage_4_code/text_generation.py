import re
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from collections import Counter
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This is my directory to find the jokes. Change accordingly to where your joke data is. 
DATA_PATH = r"C:\Users\User\OneDrive\Desktop\School\ECS 170\ECS_170\Given Stuff\Project_Stage_4\stage_4_data\text_generation\data"

SEQ_LEN = 5
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001


MODEL_TYPE = "GRU" # Change to RNN/LSTM/GRU
# Required to test all of these and mark their evaluation metrics.
# KEEP SAME AS text_classification.py


# Load Jokes

with open(DATA_PATH, "r", encoding="utf8") as f:
    text = f.read().lower()

text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

words = text.split()


# Vocabulary
counter = Counter(words)

vocab = sorted(counter.keys())

word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}


# Prepare Sequences
inputs = []
targets = []

for i in range(len(words) - SEQ_LEN):

    seq = words[i:i+SEQ_LEN]

    target = words[i+SEQ_LEN]

    inputs.append([word2idx[w] for w in seq])

    targets.append(word2idx[target])


# Dataset
class JokeDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

dataset = JokeDataset(inputs, targets)

loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True)


# Model
class TextGenerator(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 num_layers,
                 model_type="RNN"):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size,
                                      embed_dim)

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

        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.model_type = model_type

    def forward(self, x):

        x = self.embedding(x)

        out, hidden = self.rnn(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out

model = TextGenerator(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    model_type=MODEL_TYPE
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Training
loss_history = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X_batch, y_batch in tqdm(loader):

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    loss_history.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")



# Plot Loss
plt.plot(loss_history)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{MODEL_TYPE} Generator Loss")

plt.savefig("outputs/generator_loss.png")


# Generate Text
def generate_text(start_words, length=30):

    model.eval()

    current_words = start_words.lower().split()

    generated = current_words.copy()

    for _ in range(length):

        seq = current_words[-SEQ_LEN:]

        if len(seq) < SEQ_LEN:
            seq = ["the"] * (SEQ_LEN - len(seq)) + seq

        seq_idx = [word2idx.get(w, 0) for w in seq]

        x = torch.tensor([seq_idx],
                         dtype=torch.long).to(DEVICE)

        with torch.no_grad():

            output = model(x)

            predicted = torch.argmax(output, dim=1).item()

        next_word = idx2word[predicted]

        generated.append(next_word)

        current_words.append(next_word)

    return " ".join(generated)


# Example Generations
samples = [
    "what did the",
    "a man walks",
    "why did the",
    "hello my friend"
]

for s in samples:

    generated = generate_text(s)

    print("\nSTART:", s)
    print("GENERATED:", generated)

torch.save(model.state_dict(),
           f"models/{MODEL_TYPE.lower()}_generator.pth")
