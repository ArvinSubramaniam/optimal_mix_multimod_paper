#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic functions for generating data and labels and checking capacity
"""

import numpy as np
from numpy import linalg as LA
import scipy as sp
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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


def make_patterns(N,P,cod=0.5):
    matrix = np.zeros((N,P))
    for i in range(P):
        vec = generate_pm_with_coding(N,cod)
        matrix[:,i] = vec
    
    return matrix

        
def make_labels(f=0.5):
    """
    Args:
        f: refers to sparsity of labels
    """
    if np.random.rand()>1-f:
        lbl = 1
    else:
        lbl = -1
        
    return lbl


def perceptron_storage(patt,cod=0.5,kappa=0.001): #For some reason kappa needs to be non-zero but very small
    """
    To check linear classification via a linear program
    
    res.status:  = 0 if sucessfull
                != 0 otehrwise (either 1 or 2)
          
    """
    
    P = patt.shape[1]
    N = patt.shape[0]
    
    c = [0] * N
    w_bounds = (None,None)
    A = []
    B = []
     
    for m in range(patt.shape[1]):
        label=make_labels(f=cod)
        lista = list(-label/(np.sqrt(N)) * patt[:,m].T)
        A.append(lista)
        B.append(-kappa)
        
    res = linprog(c, A_ub=A, b_ub=B,bounds=w_bounds,method='revised simplex')
    
    print("status",res.status)
    print("message",res.message)
    
    return res.x, res.status
