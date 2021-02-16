#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of simplest Hebbian model
"""

from dimensionality_disentanglement import *

def compute_overlap(m,m_test):
    """
    Computes <m \tilde{m}>
    """
    N=m.shape[0]
    #P = m_test.shape[1]
    sum_ = []
    for i in range(N):
        t = m[i]*m_test[i]
        sum_.append(t)
        
    return np.mean(sum_)


"""SIMPLEST POSSIBLE HEBBIAN LEARNING"""
def simplest_possible_hebbian(N,P,d_in=0.0):
    n_real = 50
    errs = np.zeros(n_real)
    len_test = int(0.2*P)
    q_down_all = []
    q_up_all = []
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        for i in range(P):
            labels[i] = make_labels(0.5)
            #labels[i] = +1
        
        w_hebb = np.matmul(stim,labels) 
        
        four_point_list = []
        ##CREATE TEST PATTERN
        stabs = []
        patts_test = np.zeros((N,len_test))
        q_overlaps = []
        q_over_squared = []
        patts_typ = np.zeros((N,len_test))
        for n in range(len_test):#Pick 20 test points
            rand_int = np.random.randint(P)
            patt_typ = stim[:,rand_int]
            lbl_test = labels[rand_int]
            patt_test = flip_patterns_cluster(patt_typ,d_in)
            patts_test[:,n] = patt_test
            patts_typ[:,n] = patt_typ
            ov = compute_overlap(patt_typ,patt_test)
            dot_s = np.matmul(w_hebb,patt_test)
            q_overlaps.append(ov)
            q_over_squared.append(dot_s**(2))
            stab = lbl_test*np.dot(w_hebb,patt_test)
            #print("stab is",stab)
            
            stabs.append(stab)
            
            #four_point_list.append(compute_order_param_mixed(stim,patt_test))
        
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errs[j] = err

        q_up = np.mean(q_overlaps)
        q_up_all.append(q_up)

        q_down = np.mean(q_over_squared)
        q_down_all.append(q_down)

    err_mean = np.mean(errs)
    err_std = np.std(errs)
    
    q_up_mean = np.mean(q_up_all)
    numer = (1-d_in)**(2)
    
    q_down_mean = (1/N)**(2) * np.mean(q_down_all)
    q_down_final = q_down_mean - q_up_mean**(2)
    
    erf = erf1(0)
    q_theory = 1 - 8*(erf**(3) * (1-erf) + (1-erf)**(3) * erf)
    alpha = P/N
    
    denom2 = 1/N + alpha
    
    snr = (numer)/(denom2)
    snr2 = (q_up_mean**(2))/(q_down_final) #Should be exact
    err_theory = erf1(np.sqrt(snr))
    print("theoretical error is",err_theory)
    
    return err_mean, err_std, err_theory

    



