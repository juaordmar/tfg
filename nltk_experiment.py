# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:56:13 2023

@author: LENOVO
"""
import nltk
from nltk import CFG
from nltk.parse.generate import generate
from nltk.parse import RecursiveDescentParser

cfg3b = CFG.fromstring("""
    22 -> 21 20 | 20 19
    21 -> 18 16 | 16 18 17
    20 -> 16 17 | 17 16 18
    19 -> 17 18 16 | 16 17 18
    18 -> 14 13 | 15 14 13
    17 -> 15 13 14 | 14 13 15
    16 -> 13 15 14 | 15 13
    15 -> 11 12 10 | 12 11 10
    14 -> 10 11 12 | 11 10 12
    13 -> 12 11 | 11 12
    12 -> 9 7 8 | 8 9 7
    11 -> 7 8 9 | 8 7 9
    10 -> 9 8 7 | 7 9 8
    9 -> '2' '1' | '3' '2' '1'
    8 -> '3' '1' '2' | '3' '2'
    7 -> '1' '2' '3' | '3' '1'
""")

cfg3i = CFG.fromstring("""
    22 -> 21 20 19 | 19 19 20
    21 -> 18 17 | 16 16 18
    20 -> 18 18 | 17 16 17
    19 -> 16 16 | 18 16 18
    18 -> 14 15 | 14 15 13
    17 -> 15 14 | 15 15
    16 -> 14 14 | 13 13
    15 -> 11 10 12 | 11 11 10
    14 -> 10 10 | 10 10 10
    13 -> 10 12 11 | 12 11
    12 -> 8 7 | 7 9 9
    11 -> 7 7 8 | 7 7 7
    10 -> 9 9 | 8 7 7
    9 -> '1' '2' | '1' '1' '3'
    8 -> '2' '2' | '1' '1'
    7 -> '2' '3' '1' | '3' '1' '2'
""")

cfg3h = CFG.fromstring("""
    22 -> 19 21 | 20 20 21
    21 -> 17 18 17 | 17 17 18
    20 -> 17 16 | 18 16
    19 -> 18 17 | 16 17
    18 -> 14 15 15 | 15 14 14 | 15 13 13
    17 -> 15 13 15 | 13 14
    16 -> 15 13 | 14 13
    15 -> 11 11 10 | 10 12
    14 -> 12 12 10 | 10 10 | 10 12 12
    13 -> 11 10 | 12 11
    12 -> 9 8 | 8 7 | 7 9
    11 -> 7 9 9 | 7 7 | 8 7 7
    10 -> 8 8 | 9 7 | 8 7 9
    9 -> '1' '3' '3' | '2' '1' '3'
    8 -> '1' '3' | '3' '3' '1' | '1' '2'
    7 -> '1' '3' '1' | '1' '2' '3' | '2' '3' '2'
""")

cfg3g = CFG.fromstring("""
    22 -> 20 19 21 | 20 20 19 | 19 20
    21 -> 18 16 | 16 16 18 | 16 16
    20 -> 16 17 17 | 18 18 | 16 17
    19 -> 18 16 17 | 18 17 16 | 17 17 16
    18 -> 14 13 15 | 15 15 | 15 13
    17 -> 15 14 | 14 15 13 | 14 13 14
    16 -> 13 13 | 13 14 | 14 13 13
    15 -> 12 11 | 12 10 10 | 10 11
    14 -> 10 10 | 10 11 10 | 11 12
    13 -> 11 11 | 11 11 11 | 10 12
    12 -> 9 9 9 | 7 8 | 7 9
    11 -> 8 9 7 | 9 7 | 8 8 9
    10 -> 7 7 | 7 7 7 | 8 8 8
    9 -> '2' '1' | '2' '3' | '2' '3' '3'
    8 -> '3' '3' '1' | '1' '3' | '1' '3' '2'
    7 -> '2' '2' | '1' '1' | '2' '3' '1' 
""")

cfg3f = CFG.fromstring("""
    22 -> 20 20 | 21 19 19 | 20 19 21 | 20 21
    21 -> 16 18 | 16 17 18 | 17 16 | 18 17
    20 -> 17 16 18 | 16 17 | 16 16
    19 -> 18 18 | 17 18 | 18 16 18
    18 -> 13 15 | 15 13 13 | 14 15 13
    17 -> 15 14 | 14 15 | 15 14 13
    16 -> 14 14 | 14 13 | 13 15 13 | 15 15
    15 -> 12 12 11 | 10 10 | 11 11 10 | 10 11 11
    14 -> 10 12 12 | 12 11 | 12 10 12 | 10 12
    13 -> 10 12 11 | 12 11 12 | 11 12
    12 -> 8 8 9 | 9 8 | 7 9 7
    11 -> 9 7 7 | 9 7 | 8 8
    10 -> 7 9 9 | 9 7 9 | 8 9 9
    9 -> '1' '1' | '3' '3' | '1' '2' '1'
    8 -> '3' '3' '1' | '1' '2' | '3' '1' '1'
    7 -> '3' '2' | '3' '1' '2' | '3' '2' '2' | '2' '2' '1'
""")                   

# frases_cfg3b = list(generate(cfg3b, n=7000))
# out_cfg3b = ' '.join([''.join(sublist) for sublist in frases_cfg3b])

prueba = list(generate(cfg3b, n=10000))
out_cfg3b = ' '.join([''.join(sublist) for sublist in prueba])
with open('prueba.txt', 'w') as archivo:
    archivo.write(out_cfg3b)

# with open('cfg3b.txt', 'w') as archivo:
#     archivo.write(out_cfg3b)
    
# print("frases generadas y guardadas en cfg3b.txt")

# frases_cfg3i = list(generate(cfg3i, n=10000))
# out_cfg3i = ' '.join([''.join(sublist) for sublist in frases_cfg3i])

# with open('cfg3i.txt', 'w') as archivo:
#     archivo.write(out_cfg3i)
    
# print("frases generadas y guardadas en cfg3i.txt")

# frases_cfg3h = list(generate(cfg3h, n=7000))
# out_cfg3h = ''.join([''.join(sublist) for sublist in frases_cfg3h])
# print(out_cfg3h)

# with open('cfg3h.txt', 'w') as archivo:
#     archivo.write(out_cfg3h)
    
# print("frases generadas y guardadas en cfg3h.txt")

# frases_cfg3g = list(generate(cfg3g, n=7000))
# out_cfg3g = ' '.join([''.join(sublist) for sublist in frases_cfg3g])

# with open('cfg3g.txt', 'w') as archivo:
#     archivo.write(out_cfg3g)
    
# print("frases generadas y guardadas en cfg3g.txt")

# frases_cfg3f = list(generate(cfg3f, n=7000))
# out_cfg3f = ' '.join([''.join(sublist) for sublist in frases_cfg3f])

# with open('cfg3f.txt', 'w') as archivo:
#     archivo.write(out_cfg3f)
    
# print("frases generadas y guardadas en cfg3f.txt")

# The NLTK CFG implementation assumes non-terminal symbols are 
# strings, not numbers. To work with numerical symbols, you can 
# convert them to string representations.
    
# def min_length_for_cfg(cfg):
#     productions = cfg.productions()
#     min_length_dict = {}

#     for production in productions:
#         lhs = production.lhs()
#         rhs = production.rhs()

#         if isinstance(rhs[0], nltk.Nonterminal): #si rhs es un no terminal
#         #if rhs[0] == nltk.Nonterminal:
#             if lhs not in min_length_dict:
#                 min_length_dict[lhs] = 0
#             min_length_dict[lhs] = max(min_length_dict[lhs], min_length_dict.get(rhs, 0) + len(rhs))

#     start_symbol = cfg.start()
#     return min_length_dict.get(start_symbol, 0)

# isicfg = CFG.fromstring("""
#     12 -> 11 10 | 10 10 11
#     11 -> 9 8 | 8 9
#     10 -> 7 8 9 | 7 7
#     9 -> '2' '1' | '3' '2' '1'
#     8 -> '3' '1' '2' | '3' '2'
#     7 -> '1' '2' '3' | '3' '1'
# """)

# '''productions = [p for p in isicfg.productions()]
# for p in productions:
#     print(len(p.rhs()))
#     rhs = p.rhs()
#     print(rhs[0])
#     print(p.lhs())'''
# print(min_length_for_cfg(cfg3b))

