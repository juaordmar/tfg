# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:01:04 2024

@author: Juan
"""

import numpy as np
import matplotlib.pyplot as plt
 
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
 
# P = getPositionEncoding(seq_len=4, d=4, n=100)

def plotSinusoid(k, d=512, n=10000):
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2*x/d)
    y = np.sin(k/denominator)
    plt.plot(x, y)
    plt.title('i = ' + str(k))

    
def plotCosusoid(k, d=512, n=10000):
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2*x/d)
    y = np.cos(k/denominator)
    plt.plot(x, y)
    plt.title('i = ' + str(k))
    
 
fig = plt.figure(figsize=(15, 4))    
for i in range(4):
    plt.subplot(141 + i)
    plt.xlabel('x')
    plt.ylabel('sen(x)')
    plotSinusoid(i*2)
    
fig2 = plt.figure(figsize=(15, 4))    
for i in range(4):
    plt.subplot(141 + i)
    plt.xlabel('x')
    plt.ylabel('cos(x)')
    plotCosusoid(i*2)
    
# P = getPositionEncoding(seq_len=128, d=256, n=10000)
# cax = plt.matshow(P)
# plt.gcf().colorbar(cax)