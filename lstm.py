import torch
import torch.nn as nn
import numpy as np
import re
from collections import Counter
import urllib.request
import os
import tarfile
from sklearn.model_selection import train_test_split

# This class mimics the torchtext.datasets.IMDB class
class IMDBDataset:
    def __init__(self, root='.data', split='train'):
        self.root = root
        self.split = split
        self.texts = []
        self.labels = []
        
        # Download and extract if necessary
        self._download_and_extract()
        
        # Load the data
        self._load_data()
        
        # Create dataset
        self.examples = list(zip(self.labels, self.texts))
    
    def _download_and_extract(self):
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        filename = os.path.join(self.root, "aclImdb_v1.tar.gz")
        extract_path = os.path.join(self.root)
        
        # Create directory if needed
        os.makedirs(self.root, exist_ok=True)
        
        # Check if the dataset directory already exists
        imdb_path = os.path.join(self.root, "aclImdb")
        if not os.path.exists(imdb_path):
            print(f"Downloading IMDB dataset to {filename}...")
            # Download the file
            urllib.request.urlretrieve(url, filename)
            
            print("Extracting files...")
            # Extract the tar file
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall(path=extract_path)
            
            print("Dataset downloaded and extracted successfully!")
        else:
            print("Dataset already exists!")
    
    def _load_data(self):
        imdb_path = os.path.join(self.root, "aclImdb", self.split)
        
        # Process positive reviews
        pos_path = os.path.join(imdb_path, "pos")
        if os.path.exists(pos_path):
            for filename in os.listdir(pos_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as f:
                        self.texts.append(f.read())
                        self.labels.append(1)  # 1 for positive
        
        # Process negative reviews
        neg_path = os.path.join(imdb_path, "neg")
        if os.path.exists(neg_path):
            for filename in os.listdir(neg_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as f:
                        self.texts.append(f.read())
                        self.labels.append(0)  # 0 for negative
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)
    
    def __iter__(self):
        for item in self.examples:
            yield item

# Define the tokenizer (same as your original code)
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokens = text.split()
    return tokens

# Now we mimic your original code using our custom IMDB dataset

# Load datasets
train_dataset = IMDBDataset(split='train')
test_dataset = IMDBDataset(split='test')

# Create training and validation partitions
torch.manual_seed(1)
total_samples = len(train_dataset)
train_size = 20000
valid_size = 5000

# Make sure we have enough samples
if total_samples < (train_size + valid_size):
    train_size = int(0.8 * total_samples)
    valid_size = total_samples - train_size

train_dataset, valid_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, valid_size])

# Find unique tokens (words)
token_counts = Counter()
for sample in train_dataset:
    label, line = sample
    tokens = tokenizer(line)
    token_counts.update(tokens)

print('Vocab-size:', len(token_counts))

# Print the most common tokens (correctly displaying from Counter)
print('Some examples from token_counts:')
for token, count in token_counts.most_common(5):
    print(f"'{token}': {count}")

# Print some dataset examples
print("\nSome examples from the dataset:")
for i, (label, text) in enumerate(list(train_dataset)[:3]):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"Example {i+1} - {sentiment}:")
    print(text[:200] + "...\n")

# Encoding each unique token into integers
from collections import defaultdict

# Build vocab manually
token2idx = {'<pad>': 0, '<unk>': 1}
for token, _ in token_counts.most_common():
    if token not in token2idx:
        token2idx[token] = len(token2idx)

# Define a default function for unknown tokens
def encode_tokens(tokens, token2idx):
    return [token2idx.get(token, token2idx['<unk>']) for token in tokens]

# Example
print(encode_tokens(['this', 'is', 'an', 'example'], token2idx))

# Define a function for transformation
text_pipeline = lambda x: [token2idx[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x == 'pos' else 0

# Wrap the encode and transformation function
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.float32)
    lengths = torch.tensor(lengths)
    padded_text_list = nn.utils.rnn.pad_sequence(text_list, batch_first=True)
    return padded_text_list, label_list, lengths

# Take a small batch
from torch.utils.data import DataLoader
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)

text_batch, label_batch, length_batch = next(iter(dataloader))
print(text_batch)
print(label_batch)
print(length_batch)
print(text_batch.shape)

batch_size = 32
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# Embedding
embedding = nn.Embedding(num_embeddings=10, embedding_dim=3, padding_idx=0)
# a batch of 2 samples of 4 indeces each
text_encoded_input = torch.LongTensor([[1,2,4,5], [4,3,2,0]])
print(embedding(text_encoded_input))

# Building a RNN model using the nn.Module class

# class RNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
    
#     def forward(self, x):
#         _, hidden = self.rnn(x)
#         out = hidden[-1, :, :] # we use the final hidden state from the last hidden layer as the input to the fully connected layer
#         out = self.fc(out)
#         return out
    
# model = RNN(64, 32)
# print(model)
# model(torch.randn(5,3,64))

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(rnn_hidden_size*2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        out = self.embedding(text)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True)
        _, (hidden, cell) = self.rnn(out)
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
vocab_size = len(token2idx)
embed_dim = 20
rnn_hidden_size = 64
fc_hidden_size = 64
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
print(model)

def train(dataloader):
    model.train()
    total_acc, total_loss = 0, 0
    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()
        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

def evaluate(dataloader):
    model.eval()
    total_acc, total_loss = 0, 0
    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)
            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item()*label_batch.size(0)
    return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
torch.manual_seed(1)
for epoch in range(num_epochs):
    acc_train, loss_train = train(train_dl)
    acc_valid, loss_valid = evaluate(valid_dl)
    print(f'Epoch {epoch} accuracy: {acc_train:.4f} val_accuracy: {acc_valid:.4f}')

acc_test, _ = evaluate(test_dl)
print(f'test_accuracy: {acc_test:.4f}')