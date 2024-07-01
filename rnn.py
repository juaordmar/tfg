# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:42:43 2024

@author: Juan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

#hyperparameters
batch_size = 64
block_size = 16
max_iters = 5001
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
split_ratio = 0.9
rnn_hidden_size = 512
chunk_size = block_size + 1

torch.manual_seed(1337)

with open('C:/Users/Juan/Documents/ETSII Juan/5º año/TFG/ws_project/data/cfg3b.txt', 'r', encoding='utf-8') as fp: #ISO-8859-1 para español
    text = fp.read()

char_set = set(text)
chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)

text_encoded = np.array(
    [char2int[ch] for ch in text],
    dtype=np.int32)

text_chunks = [text_encoded[i:i+chunk_size]
               for i in range(len(text_encoded)-chunk_size+1)]

split_index = int(split_ratio * len(text_chunks))
train_data, val_data = text_chunks[:split_index], text_chunks[split_index:]

class TextDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self,idx): #para cada índice del dataset devuelve
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()

train_dataset = TextDataset(torch.tensor(np.array(train_data)))
val_dataset = TextDataset(torch.tensor(np.array(val_data)))
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

class RNN(nn.Module):
    def __init__(self, vocab_size, n_embd, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd) #4x384
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(n_embd, rnn_hidden_size,
                           batch_first=True) #batch_first parameter is set to True
                           #to indicate that the input data has batch size as the first dimension
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    #x: The input sequence represented as integer-encoded words
    def forward(self, x, hidden, cell): #la entrada es 1 tensor de longitud 64
        #print(x)
        out = self.embedding(x).unsqueeze(1) #(64,1,384)
       # print(out)
        out, (hidden, cell) = self.rnn(out, (hidden, cell)) #(64,1,512)
        #print(out)
        out = self.fc(out).reshape(out.size(0), -1) #(64, 4)
       # print(out)
        #reshape method is used to reshape the output tensor to have the same shape as the input tensor
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size) #(1, 64, 512)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(device), cell.to(device)

vocab_size = len(char_array)
model = RNN(vocab_size, n_embd, rnn_hidden_size)
model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(5):
    print("==================")
    print(f'Epoch {epoch}')
    print("==================")
    #Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_hidden, val_cell = model.init_hidden(batch_size)
        val_seq_batch, val_target_batch = next(iter(val_dl)) #64x16 tensors
        #cada iteración del dataloader devuelve 64 secuencias de entrada con sus
        #respectivas 64 secuencias objetivo, ambas de longitud 16
        # torch.set_printoptions(profile="full")
        # print('Val_seq_batch: \n =======================')
        # print(val_seq_batch)
        # print('Val_target_batch:')
        # print(val_target_batch)
        val_seq_batch = val_seq_batch.to(device)
        val_target_batch = val_target_batch.to(device)
        for c in range(block_size): #tengo 16 tensores de 64
            val_pred, val_hidden, val_cell = model(val_seq_batch[:, c], val_hidden, val_cell)
            # print(val_seq_batch[:, c], val_seq_batch[:, c].size())
            # print(val_target_batch[:, c], val_target_batch[:, c].size())
            val_loss += loss_fn(val_pred, val_target_batch[:, c])
        print(f'val_loss: {val_loss:.4f}')
        val_loss = val_loss.item()/block_size

    #Training
    model.train()
    loss = 0
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(train_dl))
    seq_batch = seq_batch.to(device)
    target_batch = target_batch.to(device)
    optimizer.zero_grad()
    for c in range(block_size):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])
    loss.backward()
    optimizer.step()
    print(f'loss: {loss:.4f}')
    loss = loss.item()/block_size

    # if epoch % eval_interval == 0:
    #     print(f'Epoch {epoch} - Train Loss: {loss:.4f} | Validation Loss: {val_loss:.4f}')

def sample(model, starting_str,
           len_generated_text=512,
           scale_factor=1.0):

    encoded_input = torch.tensor([char2int[s] for s in starting_str])
    encoded_input = torch.reshape(encoded_input, (1, -1))

    generated_str = starting_str

    model.eval()
    hidden, cell = model.init_hidden(1)
    hidden = hidden.to(device)
    cell = cell.to(device)
    for c in range(len(starting_str)-1):
        _, hidden, cell = model(encoded_input[:, c].view(1), hidden, cell)

    last_char = encoded_input[:, -1]
    last_char = last_char.to(device)
    for i in range(len_generated_text):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        if str(char_array[last_char]) == " ":
          return generated_str
        generated_str += str(char_array[last_char])
        last_char = last_char.to(device)

    return generated_str