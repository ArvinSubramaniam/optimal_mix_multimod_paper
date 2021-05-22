#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Context dependent pattern matrix (Barak, et. al '12)
"""

def generate_pm(N):
    v = np.zeros(N)
    for i in range(N):
        if np.random.rand() > 0.5:
            v[i] = 1
        else:
            v[i] = -1
    return v

def generate_pm_with_coding(N,f):
    """f: Fraction of plus ones"""
    v = np.zeros(N)
    for i in range(N):
        if np.random.rand() > 1-f:
            v[i] = 1
        else:
            v[i] = -1
            #v[i] = 0
    return v

def generate_pattern_context(N,M,P,K,f=0.5):
    """
    Generates an 2N x PK matrix of (input,context) pairs
    
    N: Dimension of input space
    P: Number of stimuli
    M: Number of contexts
    f: Coding level
    
    """
    mat = np.zeros((N+M,P*K))
    for p in range(P):
        stim = generate_pm_with_coding(N,f)
        for l in range(K):
            #print("index of col",p*K + l)
            mat[:N,p*K + l] = stim
    for c in range(K):
#        print("c",c)
        cont = generate_pm_with_coding(M,f)
        for k in range(P):
            mat[N:N+M,c + K*k] = cont
            #print("index of col2",c + K*k)
            
    return mat 
        

