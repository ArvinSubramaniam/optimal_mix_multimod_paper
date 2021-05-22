#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimensionality plots
"""
from perceptron_capacity_conic import *
from scipy import integrate
from random_expansion import *
from perceptron_cap_fix_rank import low_rank_reconst


gaussian_func = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**(2))

def erf1(T):
    res = integrate.quad(gaussian_func, T, np.inf)
    return res[0]

def flip(stim,pm=True):
    """
    Flip individual bit from {0,1} to {1,0}
    """
    if pm:
        if stim == -1:
            stim_o = 1
        elif stim == 1:
            stim_o = -1
        
    else:
        if stim == 0:
            stim_o = 1
        elif stim == 1:
            stim_o =0
    
    return stim_o


def flip_patterns_cluster(stim,var):
    """
    Flips other members of the cluster with prob var/2 to FORM TEST DATUM
    typ: "True' if patterns are {+1,-1}
    """
    N=stim.shape[0]
    stim_out = np.zeros(N)
    for i in range(stim.shape[0]):
        #print("i={}",i)
        if np.random.rand() > 1 - var/2:
            #print("flipped!,i={}".format(i))
            #print("stim[i]}",stim[i])
            stim_out[i] = flip(stim[i])
        else:
            stim_out[i] = stim[i]
            
    return stim_out


def compute_sparsity(stim):
    """
    Computes sparsity of patterns given
    """
    sparsity = np.sum(stim)/(len(stim))
    
    return sparsity


def compute_diff(patt_ref,patt_other):
    """
    Computes difference between reference and other pattern. Sums over neurons
    """
    return np.sum(np.abs(patt_ref - patt_other))


def compute_delta_out(out,patt_ref):
    """
    Here, out should be one vector of test pattern
    """
    
    diff = compute_diff(out,patt_ref)
                
    return (1/out.shape[0])*diff


def compute_pr_eigvals(mat):
    """
    Computes PR of a matrix using eigenvalues
    """
    eigs = LA.eigvals(mat)
    numer = np.sum(eigs)**(2)
    eigs_denom = []
    for i in range(mat.shape[0]):
        eigs_denom.append(eigs[i]**(2))
    denom = np.sum(eigs_denom)
    
    return np.real(numer/denom)

def compute_delta_out(out,patt_ref):
    """
    Here, out should be one vector of test pattern
    """
    
    diff = compute_diff(out,patt_ref)
                
    return (1/out.shape[0])*diff


def random_proj_generic_test(H,patt,test,thres,bool_=True,sparse=False):
    """
    
    Same as random_proj_generic but for both stim and test
    """
    N = patt.shape[0]
    h = np.zeros((H,patt.shape[1]))
    h_test = np.zeros((H,test.shape[1]))
    
    if sparse:
        Kd=7
        wrand = generate_random_sparse_matrix(H,N,Kd)
    else:
        wrand = np.random.normal(0,1/np.sqrt(N),(H,N))
    patt_in = patt
    test_in = test
        
    for p in range(patt.shape[1]):
        h[:,p] = np.matmul(wrand,patt_in[:,p]) - thres
        
    for q in range(test_in.shape[1]):
        h_test[:,q] = np.matmul(wrand,test_in[:,q]) - thres
        
    return h, h_test


def compute_pr_theory_sim(o,th,N,pm=False):
    """
    Args:
        o: Should be mean subtracted
        f: Sparsity
        pm: If {+,-} at mixed layer instead
    
    Returns:
        pr_emp: Empirical participation ratio
        pr_theory: Theoretical participation ratio based on 4 point corr function
        fp_corr: Four point correlation function
        
    """
    H = o.shape[0]
    P = o.shape[1]
    cov_o = np.matmul(o,o.T)
    rand_int = np.random.randint(o.shape[1])
    numer_theory = 1
    
    erf = erf1(th)
    print("erf is",erf)
    if pm:
        q1_theory = 1
        q2_theory = 1
        q3_theory = 1 - 8*(erf**(3) * (1-erf) + (1-erf)**(3) * erf)
        print("q3 is",q3_theory)
        excess_over = (16/N)*np.exp(-2*th**(2))/((2*np.pi)**(2))
        print("excess_over is",excess_over)
        q3_theory_in = q3_theory + excess_over
        
    else: 
        q1_theory = erf*(1-erf)
        q2_theory = (erf*(1-erf))**(2)
        q3_theory = (1/N) * np.exp(-2*th**(2))/((2*np.pi)**(2))
        q3_theory_in = q3_theory
        
    fp_corr_theory = q3_theory_in
    
    ratio1 = q2_theory/(q1_theory)**(2)
    print("ratio1",ratio1)
    denom_theory1 = (1/(H*P))*(ratio1) + 1/P + 1/H + (q3_theory_in/((q1_theory)**(2)))
    print("denom1_theory",denom_theory1)
    
    pr_theory = numer_theory/denom_theory1
    
    pr_emp = compute_pr_eigvals(cov_o)
    
    return pr_emp, pr_theory, fp_corr_theory



###EMPIRICALLY CALCULATE EXCESS OVERLAPS
def compute_emp_excess_over(o,H,N,th):
    """
    Empirically calculates excess overlap based on Eq (2) from Babadi,Sompolinsky

    """
    f = erf1(th)
    list_ = []
    for m in range(o.shape[1]):
        for n in range(o.shape[1]):
            if m != n:
                over = np.dot(o[:,m],o[:,n])/H
                #print("over",over)
                over2 = over - f**(2)
                list_.append(over2**(2))
                
    o_av = np.mean(list_)
    #print("o_av",o_av)
    o_av_div = o_av/(f**(2) * (1-f)**(2))
    diff =  o_av_div - 1/H
    #print("diff",diff)
    
    return N*diff
    

# N=100
# P=100
# H=2000
# thress = np.linspace(0.1,3.1,20)
# cods = np.zeros(len(thress))
# eo_emps =  np.zeros(len(thress))
# eo_theorys = np.zeros(len(thress))
# for i,th in enumerate(thress):
#     stim = make_patterns(N,P)
#     h = random_proj_generic(H,stim,th)
#     o = 0.5*(np.sign(h) + 1)
#     cod = erf1(th)
#     cods[i] = cod
#     eo = np.exp(-2*th**(2))/((2*np.pi)**(2) * cod**(2)*(1-cod)**(2))
#     eo_emp = compute_emp_excess_over(o,H,N,th)
#     eo_theorys[i] = eo
#     eo_emps[i] = eo_emp


# plt.figure()
# plt.title(r'Empirical vs theoretical $Q^{2}$',fontsize=14)
# plt.plot(cods,eo_emps,'s',color='blue',label=r'Emprical')
# plt.plot(cods,eo_theorys,'--',color='lightblue',label=r'Theory')
# plt.xlabel(r'$f$',fontsize=14)
# plt.ylabel(r'$Q^{2}$',fontsize=14)
# plt.legend()
# plt.show()


def func_sweep_cods(N,P,H,th_upper=2.8,th_lower=0.1):
    """
    Runs through different values of the threshold (sparseness) and returns 
    1. pr_emp : Empirical dimensionality
    2. pr_th : Theoretical dimensionality
    3. fp_corr : Four-point correlation
    """
    N=100
    P=200
    H=2000
    stim = make_patterns(N,P)
    stim_test = np.zeros((N,P))
    for p in range(stim.shape[1]):
        stim_test[:,p] = flip_patterns_cluster(stim[:,p],0.1)
    thress = np.linspace(th_lower,th_upper,20)
    pr_emps = np.zeros(len(thress))
    pr_theorys = np.zeros(len(thress))
    fp_corrs = np.zeros(len(thress))
    cods = np.zeros(len(thress))
    for i,th in enumerate(thress):
        print("th",th)
        #h,h_test = random_proj_generic_test(H,stim,stim_test,th)
        h = random_proj_generic(H,stim)
        o = np.sign(h-th)
        o_spars = 0.5*(o + 1)
        f = compute_sparsity(o_spars[:,np.random.randint(P)])
        o_spars_in = o_spars - f
        print("f is",f)
        cods[i] = erf1(th)
        #o_test = 0.5*(np.sign(h_test) + 1)
        pr_emp,pr_th,fp_corr = compute_pr_theory_sim(o_spars_in,th,N,pm=False)
        print("pr_theory",pr_th)
        print("pr_emp",pr_emp)
        pr_emps[i] = pr_emp
        pr_theorys[i] = pr_th
        fp_corrs[i] = fp_corr
        
    return pr_emps,pr_theorys,fp_corrs,cods

def func_find_fopt(pr_emps,cods):
    """
    Finds optimal sparseness given values of dimensionality
    """
    arg_ = np.argmax(pr_emps)
    f_opt = cods[arg_]
    return f_opt


run_dimensionality = False
if run_dimensionality:
    N=100
    P=200
    H=2000
    pr_emps,pr_theorys,fp_corrs,cods = func_sweep_cods(N,P,H)
        
#    plt.figure()
#    import matplotlib.ticker as ticker
#    ax = plt.subplot(121)
#    ax.set_title(r'Dimensionality',fontweight="bold",fontsize=16)
#    ax.plot(fp_corrs,pr_emps,'s',markersize=8,color='blue')
#    ax.plot(fp_corrs,pr_theorys,'--',color='lightblue')
#    start1, end1 = ax.get_xlim()
#    ax.set_ylabel(r'$\mathcal{D}$',fontsize=16)
#    ax.set_xlabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{\langle q_{2} \rangle^{2}}$',fontsize=16)
#    diff = fp_corrs[3] - fp_corrs[4]
#    print("start,end,diff",start1,end1,diff)
#    ax.set_xticks(np.arange(start1, end1, 3*diff))
#    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
#    
#    ax2 = plt.subplot(122)
#    ax2.set_title(r'Re-scaled interference',fontweight="bold",fontsize=16)
#    ax2.plot(cods,fp_corrs,'s-',markersize=8,color='blue')
#    start2, end2 = ax2.get_xlim()
#    diff2 = cods[1] - cods[2]
#    print("start,end,diff",start2,end2,diff2)
#    #ax.plot(fp_corrs,pr_theorys,'--',color='lightblue')
#    ax2.set_xlabel(r'$f$',fontsize=16)
#    ax2.set_ylabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{\langle q_{2} \rangle^{2}}$',fontsize=16)
#    ax2.set_xticks(np.arange(start2, end2, 3*diff2))
#    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
#    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
#    
#    plt.tight_layout()
# 
#    plt.show()
    
sweep_optimal_sparsity = False
if sweep_optimal_sparsity:
    P_list = [10,20,40,80,160,200,400]
    H_list = list(np.linspace(100,2000,4))
    N=100
    f_opts = np.zeros((len(P_list),len(H_list)))
    for i,P in enumerate(P_list):
        for j,H_in in enumerate(H_list):
            H = int(H_in)
            pr_emps,pr_theorys,fp_corrs,cods = func_sweep_cods(N,P,H)
            f_opt = func_find_fopt(pr_emps,cods)
            f_opts[i,j] = f_opt
    
    plt.figure()
    plt.title(r'Variation of $f_{opt}$ with $P$,$N_{c}$')
    colors = itertools.cycle(('green','blue','red','black'))
    for i,H_in in enumerate(H_list):
        rat = H_in/N
        clr = next(colors)
        plt.plot(P_list,f_opts[:,i],'s',color=clr,label=r'R = {}'.format(rat))
    plt.xlabel(r'$P$',fontsize=14)
    plt.ylabel(r'$f_{opt}$')
    plt.legend()
    plt.show()
    
def numer_theory(o,th):
    f = erf1(th)
    N = o.shape[0]
    P = o.shape[1]
    q2 = f*(1-f)
    q4 = q2**(2)
    numer = N**(2) * P**(2) * q4 + N * P * q4 + N * P**(2) * q4 + N**(2) * P * q4
    return numer

def denom_theory(o,th,N):
    f = erf1(th)
    H = o.shape[0]
    P = o.shape[1]
    q2 = f*(1-f)
    q4 = q2**(2)
    eo = (1/N) * np.exp(-2*th**(2))/((2*np.pi)**(2))
    denom = P*H*q4 + P * H**(2) * q4 + P**(2) * H *q4 + P**(2) * H**(2) * eo
    return denom

plot_numer_vs_denom = False
if plot_numer_vs_denom:    
    N=100
    P=100
    stim = make_patterns(N,P)
    H=2000
    ths = np.linspace(2.9,3.2,20)
    cods = np.zeros(len(ths))
    numer_emps = np.zeros(len(ths))
    denom_emps = np.zeros(len(ths))
    numer_theorys = np.zeros(len(ths))
    denom_theorys = np.zeros(len(ths))
    diff_numer = np.zeros(len(ths))
    diff_denom = np.zeros(len(ths))
    for i,th in enumerate(ths):
        h = random_proj_generic(H,stim,th)
        o = 0.5*(np.sign(h) + 1)
        f = compute_sparsity(o[:,np.random.randint(P)])
        print("f is",erf1(th))
        cods[i] = erf1(th)
        o_in = o - f
        cov = np.matmul(o_in,o_in.T)
        cov2 = np.matmul(cov,cov)
        
        numer_emp = np.matrix.trace(cov)**(2)
        print("numer_emp",numer_emp)
        denom_emp = np.matrix.trace(cov2)
        print("denom_emp",denom_emp)
        pr_emp = numer_emp/denom_emp
        print("pr_emp",pr_emp)
        numer_th = numer_theory(o_in,th)
        print("numer_theory",numer_th)
        denom_th = denom_theory(o_in,th,N)
        print("denom_theory",denom_th)
        pr_theory = numer_th/denom_th
        print("pr_theory",pr_theory)
        pr_theory2 = compute_pr_theory_sim(o_in,th,N)[1]
        
        diff_d = np.abs(denom_emp - denom_th)
        diff_n = np.abs(numer_emp - numer_th)
        
        numer_emps[i] = numer_emp
        denom_emps[i] = denom_emp
        numer_theorys[i] = numer_th
        denom_theorys[i] = denom_th
        
        diff_numer[i] = diff_n
        print("diff_n",diff_n)
        diff_denom[i] = diff_d
        print("diff_d",diff_d)
        
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title(r'Numerator')
    ax.plot(cods,numer_emps,'s',color='blue',label='Emprirical')
    ax.plot(cods,numer_theorys,'--',color='lightblue',label='Theory')
    ax.set_yscale('log')
    ax.set_xlabel(r'$f$',fontsize=14)
    ax.set_ylabel(r'$(Tr(\mathbf{C}))^{2}$',fontsize=14)
    ax.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.set_title(r'Denominator')
    ax2.plot(cods,denom_emps,'s',color='green',label='Emprirical')
    ax2.plot(cods,denom_theorys,'--',color='lightgreen',label='Theory')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$f$',fontsize=14)
    ax2.set_ylabel(r'$(Tr(\mathbf{C}^{2}))$',fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(121)
    ax.set_title(r'Difference in numerator')
    ax.plot(cods,diff_numer,'s',color='blue')
    ax.set_yscale('log')
    ax.set_xlabel(r'$f$',fontsize=14)
    ax.set_ylabel(r'Difference',fontsize=14)
    
    ax2 = fig2.add_subplot(122)
    ax2.set_title(r'Difference in denominator')
    ax2.plot(cods,diff_denom,'s',color='green')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$f$',fontsize=14)
    ax2.set_ylabel(r'Difference',fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.legend()
    plt.show()




