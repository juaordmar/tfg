# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:00:12 2023

@author: LENOVO
"""
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# batch_size = 64
# rnn_hidden_size = 512

# hidden = torch.zeros(1, batch_size, rnn_hidden_size)
# hidden.shape

# x = torch.tensor([[1, 2, 3],[2, 1, 4], [3, 4, 1], [0,2, 3]])
# print(x.shape)
# print(torch.unsqueeze(x, 1).shape)
# print(x.size(0))
# x = x.reshape(6, -1)
#print(x)
torch.cuda.is_available()

with open('C:/Users/Juan/Documents/ETSII Juan/5º año/TFG/ws_project/data/cfg3b.txt', 'r', encoding='utf-8') as fp: #ISO-8859-1 para español
    text = fp.read()
    
split_index = int(0.9 * len(text))
train_data, val_data = text[:split_index], text[split_index:]

train_tokens = train_data.split()
val_tokens = val_data.split()

train_array = np.array(train_tokens)
val_array = np.array(val_tokens)

# fig = plt.figure()  # an empty figure with no Axes
# fig, ax = plt.subplots()  # a figure with a single Axes
# plt.plot(train_array, train_array, label='linear')

def mapaTokens(tokens):
    mapa = {}
    for tok in tokens:
        if len(tok) in mapa:
            mapa[len(tok)] += 1
        else:
            mapa[len(tok)] = 1
    return mapa

print(sorted(mapaTokens(train_tokens).items()))
print(sorted(mapaTokens(val_tokens).items()))
    
        
