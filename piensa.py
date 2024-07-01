# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:35:12 2024

@author: LENOVO
"""

def minmaxLen(text):
    tokens = text.split()  # Assuming 'text' is a space-separated string
    setTokens=set(tokens)
    min_len = float('inf')
    max_len = 0
    for tok in tokens:
        tok_len = len(tok)
        if  tok_len < min_len:
            min_len = tok_len
        if tok_len > max_len:
            max_len = tok_len
    return min_len, max_len, len(tokens), len(text), len(setTokens)

cfgs = ["b", "i", "h", "g", "f"]

for cfg in cfgs:
    ruta = "C:/Users/Juan/Documents/ETSII Juan/5º año/TFG/ws_project/data/cfg3"+cfg+".txt"
    with open(ruta, "r") as f:
        text = f.read()
    print(f"Archivo: {ruta[-9:]}")
    max_len, min_len, frases_len, tokens_len, set_len  = minmaxLen(text)
    print(f"Número de tokens: {tokens_len}")
    print(f"Número de frases: {frases_len}")
    print(f"Número de frases diferentes: {set_len}")
    print(f"Longitud máxima: {max_len}")
    print(f"Longitud mínima: {min_len}")
    print("---")

# ruta = "C:/Users/Juan/Documents/ETSII Juan/5º año/TFG/ws_project/data/dataEsp.txt"
# with open(ruta, "r", encoding="ISO-8859-1") as f:
#     text = f.read()
# print(f"Archivo: {ruta}")
# max_len, min_len, frases_len, tokens_len, set_len  = minmaxLen(text)
# print(f"Número de tokens: {tokens_len/1e6:.3f}M")
# print(f"Número de frases: {frases_len}")
# print(f"Número de frases diferentes: {set_len}")
# print(f"Longitud máxima: {max_len}")
# print(f"Longitud mínima: {min_len}")
# print("---")