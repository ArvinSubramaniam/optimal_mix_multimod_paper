#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capacity of context dependent perceptron
"""

import sys
sys.path.append('/Users/arvingopal5794/Documents/masters_project/cems_model')
from fusi_barak_rank import *
from perceptron_capacity_conic import make_patterns, perceptron_storage
from numpy import linalg as LA
import random
import scipy as sp
from scipy.optimize import linprog
from scipy.special import comb
from scipy.special import binom
import itertools
from matplotlib.lines import Line2D


def theoretical_pcrit(k):
    """
    Based on simple argument, for K>2
    """
    denom = k-2
    numer = 2*(k-1)
    return numer/denom


def func_evaluate_capacity_context(K,M_list):
    
    N = 1000
    #K = 1
    #len_P = 20
    len_P = 15 #For K > 1
    n_real = 100
    sucss_matrix = np.zeros((len(M_list),len_P))
    sucss_dev = np.zeros((len(M_list),len_P))
    #P_list = np.linspace(0.2*N,3*N,len_P)
    P_list = np.linspace(1,15,15) #For K > 1
    for i,M in enumerate(M_list):
        for j,P in enumerate(P_list):
            sucs = []
            for n in range(n_real):
                patt_c = generate_pattern_context(N,M,int(P),K,f=0.5)
                w, status = perceptron_storage(patt_c)
                if status == 0:
                    sucs.append(1)
                else:
                    sucs.append(0)
            print("number in matrix",np.mean(sucs))
            sucss_matrix[i,j] = np.mean(sucs)
            sucss_dev[i,j] = np.std(sucs)

    plt.figure()
    plt.title(r'CDP capacity, K={}'.format(K),fontsize=16)
    for i,M in enumerate(M_list):
        ind_up = np.where(sucss_matrix[i,:] + sucss_dev[i,:] >= 1.)[0] #Check if it's too high
        sucss_dev[i,ind_up] = np.ones(len(ind_up)) - sucss_matrix[i,ind_up]
        
        ind_down = np.where(sucss_matrix[i,:] - sucss_dev[i,:] <= 0.)[0]#Check if it's too low
        sucss_dev[i,ind_down] = sucss_matrix[i,ind_down]
        
        plt.errorbar(P_list,sucss_matrix[i,:],yerr=sucss_dev[i,:],marker='s',linestyle='-', capsize=5, markeredgewidth=2,
                     label=r'$M={}$'.format(M))
    if K != 2 and False:
        ptheory = np.round(theoretical_pcrit(K),2)
        plt.axvline(x=ptheory,linestyle='dashdot',color='r',label=r'Theoretical $P_c = {}$'.format(ptheory))
    plt.axvline(x=2,linestyle='dashdot',color='r',label=r'Cover $P_c = 2$')
    plt.axhline(y=0.5,linestyle='--',label='Prob. = 0.5')
    plt.xlabel(r'$P$',fontsize=14)
    plt.ylabel('Prob of success',fontsize=14)
    plt.legend(fontsize=12)
    plt.savefig(r'{}/capacity_curve_errorbars_K={}.png'.format(path,K))
    plt.show()
    
    #Find Pcrit
    find_Pcrit = False
    if find_Pcrit:
        Pcrits = []
        for i,M in enumerate(M_list):
            ind1 = set(np.where(sucss_matrix[i,:]<=0.6)[0])
            print("ind1",ind1)
            ind2 = set(np.where(sucss_matrix[i,:]>=0.4)[0])
            print("ind2",ind2)
            ind_ = list(ind2.intersection(ind1))
            if len(ind_) == 0:
                p1 = P_list[list(ind1)[0]]
                p2 = P_list[list(ind2)[-1]]
                pmean = np.mean([p1,p2])
                Pcrits.append(pmean)
            else:
                ind_med = int(np.median(ind_))
                Pcrits.append(P_list[ind_med])
        
        return Pcrits[0]

        
        
def check_theory_context1(N,M,P,K):
    """
    Gives a theoretical check on the capacity, by comparing the rank using Kabashima's result
    """
    if N > P and M > K:
        rank = P + K - 1
    alpha = (P*K)/(N+M)
    print("alpha, rank are {},{}".format(alpha,rank))
    print("val",2/(N+M) * rank)
    if alpha > (2/(N+M)) * rank:
        return False
    else:
        return True
    
def check_theory_context_thermo(N,M,P,K):
    """
    Gives a theoretical check on the capacity, for large P and N
    """
    beta = P/N
    delta = M/N
    alpha = (P*K)/(N+M)
    cap = 2*(beta)/(1+delta)
    print("alpha is {}, cap is {}".format(alpha,cap))
    if alpha > cap:
        return False
    else:
        return True
    

def context_matrix_calc_corr():
    """
    Feasability check for different c1, c2
    """
    N = 100
    M = 100
    P = 90
    K = 1
    n_real = 5
    
    c1_list = np.linspace(0.1,0.5,8)
    c2_list = np.linspace(0.1,0.5,8)
    success_matrix = np.zeros((len(c1_list),len(c1_list)))
    for i,C1 in enumerate(c1_list):
        for j,C2 in enumerate(c2_list):
            counts = []
            for n in range(n_real):
                patt = generate_pattern_context2(N,M,int(P),int(K),C1,C2)
                res = check_theory_context1(N,M,int(P),int(K))
                print("result should be",res)
                print("c1,c2 are",C1,C2)
                w, status = perceptron_storage(patt,kappa=0.01)
                if status == 0:
                    counts.append(1)
                else:
                    counts.append(0)
            success_matrix[i,j] = np.mean(counts)
            
    fig = plt.figure(figsize=[5.0,5.0])
    ax = fig.add_subplot(111) 
    plt.title(r'Success matrix, $P={}$,$K={}$'.format(P,K))
    print("shape matrix",success_matrix.shape)
    plt.imshow(success_matrix)
    ax.set_xticklabels(np.round(c1_list,2))
    ax.set_yticklabels(np.round(c2_list,2))
    plt.xlabel(r'$c_{\xi}$')
    plt.ylabel(r'$c_{\phi}$')
    #plt.colorbar()
    plt.show()
    

def context_matrix_calc(cod=0.5):
    """
    Feasability check for different P,K
    """
    N = 50
    M = 50
    n_real = 30
    len_P = 20
    len_K = 5
    P_list = np.linspace(1,len_P,len_P)
    K_list = np.linspace(1,len_K,len_K)
    success_matrix = np.zeros(((len_P),(len_K)))
    for i,P in enumerate(P_list):
        for j,K in enumerate(K_list):
            counts = []
            for n in range(n_real):
                patt = generate_pattern_context(N,M,int(P),int(K),f=cod)
                res = check_theory_context1(N,M,int(P),int(K))
                print("result should be",res)
                w, status = perceptron_storage(patt,cod_l=cod)
                if status == 0:
                    counts.append(1)
                else:
                    counts.append(0)
            success_matrix[i,j] = np.mean(counts)
            
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(r'Success matrix, $f={}$'.format(cod))
    plt.imshow(success_matrix)
    ax.set_xticks(np.arange(len_K))
    ax.set_yticks(np.arange(len_P))
    ax.set_xticklabels([int(k) for k in K_list])
    ax.set_yticklabels([int(p) for p in P_list])
#    plt.xticks(K_list)
#    plt.yticks(P_list)
    plt.xlabel(r'$K$',fontsize=14)
    plt.ylabel(r'$P$',fontsize=14)
    plt.colorbar()
    plt.show()
    
def func_evaluate_pcrit_diff_k_n(f=0.5):
    """
    Function that evaluates Pcrit for different K and N
    """
    N_list = [25,32,50,65,80]
    #N_list = [40,80]
    #len_K = 5
    len_P = 20
    n_real = 30
    #K_list = np.linspace(1,len_K,len_K)
    K_list = [2,3,4]
    len_K = len(K_list)
    Pcrits = np.zeros((len(N_list),len_K))
    Pdevs = np.zeros((len(N_list),len_K))
    for i, N in enumerate(N_list):
        M = N
        P_list2 = list(np.linspace(0.5*N,3.0*N,len_P))
        P_list1 = list(K_list)
        P_list = P_list1 + P_list2
        for j,K in enumerate(K_list):
            for n in range(n_real):
                Pcrit_list = []
                for k,P in enumerate(P_list):
                    patt = generate_pattern_context(N,M,int(P),int(K))
                    #patt = make_patterns(N,int(P),cod=f)
                    w, succ = perceptron_storage(patt,cod_l=f)
                    if succ==0:
                        print("success, for P",int(P))
                        None
                    else:
                        Pcrit_list.append(P)
                        print("Break!, Pcrit is",P,"K=",K)
                        break
                        
            Pcrit = np.mean(Pcrit_list)
            Pdev = np.std(Pcrit_list)
            Pcrits[i,j] = Pcrit
            Pdevs[i,j] = Pdev
            
    plt.figure(1)
    plt.title(r'$P_c$ vs. $K$ for different N',fontsize=18)
    for i,N in enumerate(N_list):
        print("devs",Pdevs[i,:])
        plt.errorbar(K_list,Pcrits[i,:],yerr = Pdevs[i,:],capsize=2.0,marker='s',label=r'$N={}$'.format(N))
    plt.xlabel(r'$K$',fontsize=14)
    plt.ylabel(r'$P_c$',fontsize=14)
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.title(r'$P_c$ vs. $N$ for different K',fontsize=18)
    y = [2*e for e in N_list]
    #print("y",y)
    for i,K in enumerate(K_list):
        print("devs",Pdevs[:,i])
        plt.errorbar(N_list,Pcrits[:,i],yerr = Pdevs[:,i],capsize=2.0,marker='o',label=r'$K={}$'.format(K))
    plt.plot(N_list,y,linestyle='--',label=r'Theory')   
    plt.xlabel(r'$N$',fontsize=14)
    plt.ylabel(r'$P_c$',fontsize=14)
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title(r'$P_c$ for different $K$ and $N$',fontsize=18)
    plt.imshow(Pcrits)
    ax.set_yticks(np.arange(len(N_list)))
    ax.set_xticks(np.arange(len_K))
    ax.set_yticklabels([int(n) for n in N_list])
    ax.set_xticklabels([int(k) for k in K_list])
    plt.xlabel(r'$K$',fontsize=14)
    plt.ylabel(r'$N$',fontsize=14)
    plt.colorbar()
    plt.show()
    
    return Pcrits

def run_cover_first():
    N_list=[100]
    N = N_list[0]
    K_list = [2,3,4] #FINITE P
    #K = 1
    #M_list = [10,80,100,150]
    M_list = [100] #Finite P
    #P_list = np.linspace(1,3*N,3*N)
    P_list = np.linspace(1,10,10) #FINITE P
    cs = {}  
    
    for i, K in enumerate(K_list):
    #for i, M in enumerate(M_list):
        cs[i] = []
        for j,P in enumerate(P_list):
            c_list = []
            
            top_sum = N_list[0] + M_list[0]
            top_c = P + K - 1
            bott_exp = P*K
            
            for k in range(top_sum):
                c_list.append(comb(top_c - 1,k))
            c = 2*np.sum(c_list)
            cs[i].append(c/(2**(bott_exp)))
        
    plt.figure()
    markers = itertools.cycle(('s', '^', 'o','D'))
    #plt.title(r'Cover theorem - CDP, $K={}$,$N={}$'.format(K,N),fontsize=18)
    plt.title(r'Cover theorem - CDP, $M={}$,$N={}$'.format(M_list[0],N),fontsize=18)
    #for i,M in enumerate(M_list):
    for i,K in enumerate(K_list):
        plt.plot(P_list,cs[i],marker=next(markers),label=r'$K={}$'.format(K)) #FINITE P
        #plt.plot((1/N)*P_list,cs[i],marker=next(markers),label=r'$\delta={}$'.format(M/N))
    #plt.xlabel(r'$\beta = \frac{P}{N}$',fontsize=14)
    plt.xlabel(r'$P$',fontsize=14) #FINITE P
    plt.ylabel('Prob',fontsize=14)
    plt.legend(fontsize=12)
    plt.show()
    
    
run_cover_second = False
if run_cover_second:
    """
    Speculative probablistic Cover's theorem
    """
    N_list=[100]
    N = N_list[0]
    #K_list = [2,3,4,5] #FINITE P
    K = 1
    M_list = [10,80,100,150] 
    P_list = np.linspace(1,3*N,3*N)
    #P_list = np.linspace(1,10,10) #FINITE P
    cs = {}  
    
    #for i, K in enumerate(K_list):
    for i, M in enumerate(M_list):
        cs[i] = []
        for j,P in enumerate(P_list):
            c_list1 = []
            c_list2 = []
            d_list1 = []
            d_list2 = []
            
            coeff1 = (K/(P+K-1))
            print("coeff 1",coeff1)
            #coeff1 = 1
            coeff2 = ((P-1)/(P+K-1))
            print("coeff 2",coeff2)
            #coeff2 = 0
            
            #top_sum1 = N_list[0] + 50 #FINITE P
            top_sum1 = N_list[0] + M
            top_sum2 = N_list[0]
            
            top_c1 = K
            top_c2 = P - 1
            
            bott_exp1 = P + K - 1
            bott_exp2 = P + K - 1
            
            for k1 in range(top_sum1):
                c_list1.append(comb(top_c1 - 1,k1))
            for k2 in range(top_sum2):
                c_list2.append(comb(top_c2 - 1,k2))
                
            c1 = (2*np.sum(c_list1))/(2**(bott_exp1))
            print("first contribtution {}, P={}, K={}".format(c1,P,K))
            c2 = (2*np.sum(c_list2))/(2**(bott_exp2))
            print("second contribtution {}, P={}, K={}".format(c2,P,K))
            c = coeff1*c1 + coeff2*c2
            
            ###TO ENFORCE SYMMETRY OF PROBLEM??
            coeff1d = (P/(P+K-1))
            print("coeff 1",coeff1)
            #coeff1 = 1
            coeff2d = ((K-1)/(P+K-1))
            print("coeff 2",coeff2)
            #coeff2 = 0
            
            #top_sum1d = N_list[0] + 50 #FINITE P
            top_sum1 = N_list[0] + M
            top_sum2d = 50
            
            top_c1d = P
            top_c2d = K- 1
            
            bott_exp1d = P + K - 1
            bott_exp2d = P + K - 1
            
            for k1 in range(top_sum1d):
                d_list1.append(comb(top_c1d - 1,k1))
            for k2 in range(top_sum2d):
                d_list2.append(comb(top_c2d - 1,k2))
                
            d1 = (2*np.sum(d_list1))/(2**(bott_exp1d))
            print("first contribtution {}, P={}, K={}".format(d1,P,K))
            d2 = (2*np.sum(d_list2))/(2**(bott_exp2d))
            print("second contribtution {}, P={}, K={}".format(d2,P,K))
            d = coeff1d*d1 + coeff2d*d2
            
            cont1 = 0.5
            print("cont1",cont1)
            cont2 = 0.5
            print("cont2",cont2)
            
            tot = cont1*c + cont2*d
            
            cs[i].append(tot)
        
    plt.figure()
    markers = itertools.cycle(('s', '^', 'o','D'))
    plt.title(r'Cover theorem - CDP, $K={}$,$N={}$'.format(K,N),fontsize=18)
    #plt.title(r'Cover theorem - CDP, $M={}$,$N={}$'.format(100,N),fontsize=18)
    for i,M in enumerate(M_list):
    #for i,K in enumerate(K_list):
        #plt.plot(P_list,cs[i],marker=next(markers),label=r'$K={}$'.format(K)) #FINITE P
        plt.plot((1/N)*P_list,cs[i],marker=next(markers),label=r'$\delta={}$'.format(M/N))
    plt.xlabel(r'$\beta = \frac{P}{N}$',fontsize=14)
    #plt.xlabel(r'$P$',fontsize=14) #FINITE P
    plt.ylabel('Prob',fontsize=14)
    plt.legend(fontsize=12)
    plt.show()
    



    