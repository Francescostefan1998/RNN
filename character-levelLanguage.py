import numpy as np
# Reading and processing text
with open('1268-0.txt', 'r', encoding="utf-8") as fp:
    text=fp.read()

start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')
text = text[start_indx:end_indx]
char_set = set(text)
print('start index: ', start_indx)
print('end index: ', end_indx)
print('Total Length:', len(text))
print('Unique Characters:', len(char_set))

# We convert the character into integer and vice versa
# Building the dictionary
chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print('Text encoded shape:', text_encoded.shape)
print(text[:15], '== Encoding ==>', text_encoded.shape)
print(text_encoded[15:21], '== Reverse ==>', ''.join(char_array[text_encoded[15:21]]))
for ex in text_encoded[:5]:
    print('{} -> {}'.format(ex, char_array[ex]))

from torch.utils.data import Dataset
seq_length = 40
chunk_size = seq_length + 1
text_chunks = [text_encoded[i:i + chunk_size] for i in range(len(text_encoded)-chunk_size+1)]
from torch.utils.data import Dataset
class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunks = self.text_chunks[idx]
        return text_chunks[:-1].long(), text_chunks[1:].long()
    
import torch
seq_dataset = TextDataset(torch.tensor(text_chunks))

# Take a look at some example sequences from this transformed dataset
for i, (seq, target) in enumerate(seq_dataset):
    print(' Input (x): ', repr(''.join(char_array[seq])))
    print('Target (y): ', repr(''.join(char_array[target])))
    print()
    if i == 1:
        break

# Transform this dataset into mini-batches
from torch.utils.data import DataLoader
batch_size = 64
torch.manual_seed(1)
seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell
    
vocab_size = len(char_array)
embed_dim = 256
rnn_hidden_size = 512
torch.manual_seed(1)
model = RNN(vocab_size, embed_dim, rnn_hidden_size)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epochs = 50
torch.manual_seed(1)
for epoch in range(num_epochs):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    loss = loss.item()/seq_length
    if epoch % 10 == 0:
        print(f'Epoch {epoch} loss: {loss:.4f}')


# Evaluation phase
from torch.distributions.categorical import Categorical
torch.manual_seed(1)
logits = torch.tensor([[1.0, 1.0, 1.0]])
print('Probabilities:', nn.functional.softmax(logits, dim=1).numpy()[0])
m = Categorical(logits=logits)
samples = m.sample((10,))
print(samples.numpy())

torch.manual_seed(1)
logits = torch.tensor([[1.0, 1.0, 3.0]])
print('Probabilities:', nn.functional.softmax(logits, dim=1).numpy()[0])
m = Categorical(logits=logits)
samples = m.sample((10,))
print(samples.numpy())

def sample(model, starting_str, len_generated_text = 500, scale_factor=1.0):
    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))
    generated_str = starting_str
    model.eval()
    hidden, cell = model.init_hidden(1)
    for c in range(len(starting_str)-1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)

    last_char = encoded_input[:, -1]
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        generated_str += str(char_array[last_char])

    return generated_str

torch.manual_seed(1)
print(sample(model, starting_str='The Island'))