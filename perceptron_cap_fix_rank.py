#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perceptron capacity - fix rank (KABSHIMA) - correlated patterns
"""

import sys
#sys.path.append('/Users/arvingopal5794/Documents/cognitive_control/context_rank_spectra')
from fusi_barak_rank import *
from perceptron_capacity_conic import *
from numpy import linalg as LA
import random
import scipy as sp
from scipy.optimize import linprog
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def func_evaluate_capacity_fixrank(f=0.5):
    """
    For different P and N, calculate alpha_c for different c's
    Fix c < 0.5*min(P,N) such that can reach capacity
    """
    
    N_list = [100]
    len_P =  8
    n_real = 10
    for i,N in enumerate(N_list):
        P_list = np.linspace(0.5*N,3*N,len_P)
        P_crits = {}
        P_crits_dev = {}
        c_lists = {}
        for j,P in enumerate(P_list):
            len_c = 5
            P_crits[j] = []
            P_crits_dev[j] = []
            P_crits_full = np.zeros((n_real,len_c))
            for n in range(n_real):
                patt = make_patterns(N,int(P),cod=f)
                c_max = (1/N)*0.5*min(P,N)
                c_lists[j] = np.linspace(0.1,c_max,len_c)
                print("c_list",c_lists[j])
                for k,C in enumerate(c_lists[j]):
                    patt_c = low_rank_reconst(patt,C)
                    c = int(C*N)
                    Ps = np.linspace(0.1*P,P,20)
                    for p in Ps:
                        patt_in = patt_c[:,:int(p)]
                        print("checking")
                        w,stat = perceptron_storage(patt_in,cod_l=f)
                        if stat == 0:
                            None
                        else:
                            P_crits_full[n,k] = p
                            print("appending Pcrit",p,"for c,P",c,int(P))
                            break
                        
            P_crits[j] = np.mean(P_crits_full,axis=0) #list of length c
            P_crits_dev[j] = np.std(P_crits_full,axis=0)
            print("full P_crit list",P_crits[j],"for c = {}".format(c))
            print("full P_crit std",P_crits_dev[j],"for c = {}".format(c))
            
        
        #PLOT PCRITS for each (P,N)
        plt.figure()
        plt.title(r'Plot of $P_c$, $N={}$, $f={}$'.format(N,f),fontsize=18)
        for j,P in enumerate(P_list):
            print("Pcrit list",P_crits[j])
            plt.errorbar(c_lists[j], P_crits[j], yerr=P_crits_dev[j],label=r'$P={}$'.format(int(P)))
        y = 2*c_lists[len_P-1]*N
        plt.plot(c_lists[len_P-1],y,'--',label=r'theory,$P_c = 2cN$')
        plt.xlabel(r'$c$',fontsize=16)
        plt.ylabel(r'$P_c$',fontsize=16)
        plt.legend(fontsize=12)
        plt.show()
        

def func_evaluate_capacity_fixrank_varycoding():
    """
    Same function as above, but vary coding level
    """
    
    N_list = [50]
    len_P =  8
    n_real = 10
    f_list=  [0.001,0.5,0.999]
    #f_list = [0.98]
    for i,N in enumerate(N_list):
        P = 2*N
        P_crits = {}
        P_crits_dev = {}
        c_lists = {}
        for j,f in enumerate(f_list):
            len_c = 5
            P_crits[j] = []
            P_crits_dev[j] = []
            P_crits_full = np.zeros((n_real,len_c))
            for n in range(n_real):
                patt = make_patterns(N,int(P),cod=f)
                c_max = (1/N)*0.5*min(P,N)
                c_lists[j] = np.linspace(0.1,c_max,len_c)
                print("c_list",c_lists[j])
                for k,C in enumerate(c_lists[j]):
                    patt_c = low_rank_reconst(patt,C)
                    c = int(C*N)
                    Ps = np.linspace(0.1*P,P,20)
                    for p in Ps:
                        patt_in = patt_c[:,:int(p)]
                        print("checking")
                        w,stat = perceptron_storage(patt_in,cod_l=f)
                        if stat == 0:
                            None
                        else:
                            P_crits_full[n,k] = p
                            print("appending Pcrit",p,"for c,f",c,f)
                            break
                        
            P_crits[j] = np.mean(P_crits_full,axis=0) #list of length c
            P_crits_dev[j] = np.std(P_crits_full,axis=0)
            print("full P_crit list",P_crits[j],"for c = {}".format(c))
            print("full P_crit std",P_crits_dev[j],"for c = {}".format(c))
            
        
        #PLOT PCRITS for each (P,N)
        plt.figure()
        plt.title(r'Plot of $P_c$, $N={}$, $P={}$'.format(N,P),fontsize=18)
        for j,f in enumerate(f_list):
            print("Pcrit list",P_crits[j])
            plt.errorbar(c_lists[j], P_crits[j], yerr=P_crits_dev[j],label=r'$f={}$'.format(f))
        len_f = len(f_list)
        y = 2*c_lists[len_f-1]*N
        plt.plot(c_lists[len_f-1],y,'--',label=r'theory,$P_c = 2cN$')
        plt.xlabel(r'$c$',fontsize=16)
        plt.ylabel(r'$P_c$',fontsize=16)
        plt.legend(fontsize=12)
        plt.show()
          
    
    
def capacity_check(c,patt,len_P,cod_out=0.5):
    """
    Calculates capacity for given pattern and c
    Tries P values in the interval (0.5*N,2*N,len_P)
    """
    patt_c = low_rank_reconst(patt,c)
    print("patt_c shape",patt_c.shape)
    print("patt_c rank",LA.matrix_rank(patt_c))
    P_list = np.linspace(0.1*patt.shape[1],1*patt.shape[1],len_P)
    for i,P in enumerate(P_list):
        patt_in = patt_c[:,:int(P)]
        w, succ = perceptron_storage(patt_in,cod_l=cod_out)
        if succ == 0:
            None
        else:
            Pcrit = P
            print("Pcrit is",Pcrit)
            break
    return Pcrit
    
    
    
def make_patterns_correlated(N,P,fc=0.9,cod=0.5):
    """
    NOT NEEDED?
    """
    ind_rem = random.sample(range(P), int((1-fc)*N))
    print("rank should be",int(fc*N))
    
    patterns = make_patterns(N,P)
    patterns_red = np.copy(patterns)
    
    #Make a repeated pattern vector
    vec = generate_pm_with_coding(N,cod)
    #Loop over distinct pairs of (remove,replace)
    for i in ind_rem:
        patterns_red[:,i] = vec
     
    print("rank original",LA.matrix_rank(patterns))
    print("rank reduced",LA.matrix_rank(patterns_red))
    
    return patterns_red
    
def low_rank_reconst(patt,C=0):
    """
    USE THIS
    
    """
    c = int(C*patt.shape[0])
    u, d, v = np.linalg.svd(patt)
    
    if c == 0:
        eigs_red = np.delete(d,np.where(d <= 10e-10)[0])
    else:
        eigs_red = d[:c]
        
    print("length of eigs reduced",len(eigs_red))
     
    D = np.zeros((patt.shape[0],patt.shape[1]))
    if patt.shape[0] >= patt.shape[1]:
        for i in range(len(eigs_red)):
            D[i,i] = eigs_red[i]
    elif patt.shape[0] < patt.shape[1]:
        for i in range(len(eigs_red)):
            D[i,i] = eigs_red[i]
    
#    plt.figure()
#    plt.title("D matrix")
#    plt.imshow(D)
#    plt.colorbar()
#    plt.show()
      
    patt_r = np.matmul(u,np.matmul(D,v))
    
    return patt_r

def plot_patterns():

    N=100
    P=80
    
    patt1 = make_patterns(N,P,cod=0.5)
    patt1_r = low_rank_reconst(patt1,C=0.5)
    patt2 = make_patterns(N,P,cod=0.9)
    patt2_r = low_rank_reconst(patt2,C=0.5)
    
    fig = plt.figure(figsize=[7.0,7.0])
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(221)
    ax1.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax2 = fig.add_subplot(223)
    ax2.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax3 = fig.add_subplot(222)
    ax3.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax4 = fig.add_subplot(224)
    ax4.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    
    #Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    
    ax1.imshow(patt1)
    ax1.set_title(r'$c={}$,$f={}$'.format(1.0,0.5))
    ax2.imshow(patt1_r)
    ax2.set_title(r'$c={}$,$f={}$'.format(0.5,0.5))
    ax3.imshow(patt2)
    ax3.set_title(r'$c={}$,$f={}$'.format(1.0,0.9))
    ax4.imshow(patt2_r)
    ax4.set_title(r'$c={}$,$f={}$'.format(0.5,0.9))
    
    ax.set_xlabel(r'$f$',fontsize=18)
    ax.set_ylabel(r'$c$',fontsize=18)
    plt.show()
    



