#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparseness and expansion - result of unimodal model
"""

from perceptron_capacity_conic import *
from random_expansion import *
from hebbian_readout import *
from dimensionality_disentanglement import *
from scipy import integrate
import itertools

gaussian_func = lambda x: (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**(2))

def erf1(T):
    res = integrate.quad(gaussian_func, T, np.inf)
    #print("IN ERR FUNC!")
    return res[0]

###ORDER OF X,Y IMPORTANT!
gaussian_func_2dim = lambda y,x: (1/np.sqrt(2*np.pi))*np.exp(-0.5*x**(2)) * (1/np.sqrt(2*np.pi))*np.exp(-0.5*y**(2)) 

def lower_bound(T,ds,x):
    b = ((1-ds)*x - T)/(np.sqrt(ds*(2-ds)))
    return b

def erf_full(T,ds,f):
    res = integrate.dblquad(gaussian_func_2dim, T, np.inf, lambda x: lower_bound(T,ds,x), lambda x: np.inf)
    return 1/(f*(1-f)) * res[0]

def erf_full1(T,ds):
    res = integrate.dblquad(gaussian_func_2dim, T, np.inf, lambda x: lower_bound(T,ds,x), lambda x: np.inf)
    return res[0]


"""Generate \Delta m plots"""
def generate_delta_m(N,P,H,d_in,th,pm=True):
    n_real = 1
    len_test = int(0.2*P)
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        patts_test = np.zeros((N,len_test))
        labels_test = []
        ints = []
        ##CREATE TEST PATTERNS
        for n in range(len_test):#Pick test points - perturb ONE pattern randomly
            rand_int = np.random.randint(P)
            patt_typ = stim[:,rand_int]
            lbl_test = labels[rand_int]
            labels_test.append(lbl_test)
            patt_test = flip_patterns_cluster(patt_typ,d_in)
            d_in_check = compute_delta_out(patt_test,patt_typ)
            print("check d_in",d_in_check)
            patts_test[:,n] = patt_test
            ints.append(rand_int)
        
        print("before random projection")
        h,h_test = random_proj_generic_test(H,stim,patts_test,th,bool_=pm)
        print("after random projection")
        
        o_spars = 0.5*(np.sign(h)+1)
        o = np.sign(h)
        o_test = np.sign(h_test)
        f = compute_sparsity(o_spars[:,np.random.randint(P)])
        print("coding",f)
        
        o_test_spars = 0.5*(np.sign(h_test)+1)
        
        if pm==True:
            o_in = o
            o_test_in = o_test
            
        else:
            o_in = o_spars
            o_test_in = o_test_spars
        
        w_hebb = np.matmul(o_in,labels) 
        
        erf = erf1(th)
        
        #print("shape hebbian weights",w_hebb.shape)
        stabs = []
        d_outs = []
        acts_typ = np.zeros((H,len_test))
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            stabs.append(stab)
            if pm:
                d_out = compute_delta_out(o_test_in[:,m],o_in[:,ints[m]])
            else:
                d_out = (1/(2*erf*(1-erf))) * compute_delta_out(o_test_in[:,m],o_in[:,ints[m]])
            d_outs.append(d_out)
            acts_typ[:,m] = o_in[:,ints[m]]
        
        d_out_mean = np.mean(d_outs)
        d_std = np.std(d_outs)
        print("d_out_mean",d_out_mean)
        
        if pm:
            d_out_theory = 4*erf_full1(th,d_in)
        else:
            d_out_theory = erf_full(th,d_in,erf)
        
    return d_out_mean, d_out_theory, erf

# N=100
# P=100
# H=2100
# th=0.8
# ds=0.1
# d_emp,d_theory,f=generate_delta_m(N,P,H,ds,th)
# print("d_emp",d_emp,"d_theory",d_theory)


def excess_over_theory(th,f):
    numer = np.exp(-th**(2))
    denom = 2*np.pi*f*(1-f)
    return numer/denom

def excess_over_no_f(th):
    numer = np.exp(-th**(2))
    denom = 2*np.pi
    return numer/denom

    
###CHECK DIMENSIONALITY VS. EO
plot_dim = False
if plot_dim:
    bool_=False
    H_list = [50,100,200]
    ths = np.linspace(0.02,2.1,10)
    #ths = [0.8]
    n_trials=1
    pr_theorys = np.zeros(len(ths))
    pr_emps = np.zeros(len(ths))
    pr_emps_dev = np.zeros(len(ths))
    fp_corr_means = np.zeros(len(ths))
    eo_means = np.zeros(len(ths))
    cods = np.zeros(len(ths))
    for i,th in enumerate(ths):
        fp_corrs = []
        pr_emps_trials = []
        pr_theorys_trials = []
        eo_trials = []
        N=100
        M=100
        P=1000
        H=1200
        for n in range(n_trials):
            stim = make_patterns(N,P)
            cont = make_patterns(M,K)
            #cont = np.zeros((M,K))
            #h = random_project_hidden_layer(stim,cont,H) - th #NOT NORMALIZED!!
            h = random_proj_generic(H,stim,thres=th)
            if bool_:
                o = np.sign(h)
            else:
                o = 0.5*(np.sign(h)+1)
            o_spars = 0.5*(np.sign(h)+1)
            f = compute_sparsity(o_spars[:,np.random.randint(P)])
            o_in = o - f
            f_in = erf1(th)
            cods[i] = erf1(th)
            pr_emp, pr_theory, fp_corr = compute_pr_theory_sim(o_in,th,N,pm=bool_)
            eo = excess_over_theory(th,f_in)
            eo_in = (eo**(2))/(N)
            pr_theory_eo = 1/(1/(H*P) + (1/P) + (1/H) + eo_in)
            pr_theory_eo2 = 1/(eo_in)
            print("pr_emp",pr_emp)
            print("pr_theory",pr_theory)
            print("pr_theory_eo",pr_theory_eo)
            pr_emps_trials.append(pr_emp)
            pr_theorys_trials.append(pr_theory_eo)
            fp_corrs.append(fp_corr)
            eo_trials.append(eo_in)
    
        pr_emp_mean = np.mean(pr_emps_trials) 
        pr_emp_std = np.std(pr_emps_trials) 
        pr_theory_mean = np.mean(pr_theorys_trials)
        fp_corr_mean = np.mean(fp_corrs)
        eo_mean = np.mean(eo)
        print("averaged correlation",fp_corr_mean)
        print("eo_mean",eo_mean)
        
        pr_theorys[i] = pr_theory_mean
        pr_emps[i] = pr_emp_mean
        pr_emps_dev[i] = pr_emp_std
        fp_corr_means[i] = fp_corr_mean
        eo_means[i] = eo_mean
        
        
    plt.figure()
    plt.title(r'$\mathcal{D}$ vs. $Q^{2} - $unimodal$ ($P=1000$, $H=1200$)$',fontsize=12)
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    clr = next(colors)
    clr_theory = next(colors_ver)
    plt.errorbar(eo_means,pr_emps,yerr = pr_emps_dev,color=clr,fmt='s', 
                 capsize=5, markeredgewidth=2)
    plt.plot(eo_means,pr_theorys,'--',color=clr_theory)
    #plt.plot(cods,fp_corr_means,'s',color=clr)
    #plt.plot(cods,eo_means,'s-',color=clr)
    plt.ylabel(r'$\mathcal{D}$',fontsize=14)
    plt.xlabel(r'$Q^{2}$',fontsize=14)
    plt.legend()
    plt.show()
    #
    #
    plt.figure()
    plt.title(r'$Q^{2}$ vs.$T$ - $unimodal$ ($P=1000$, $H=1200$)',fontsize=12)
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    clr = next(colors)
    clr_theory = next(colors_ver)
    #plt.errorbar(eo_means,pr_emps,yerr = pr_emps_dev,color=clr,fmt='s', 
                 #capsize=5, markeredgewidth=2)
    #plt.plot(cods,fp_corr_means,'s',color=clr)
    plt.plot(cods,eo_means,'s-',color=clr)
    plt.ylabel(r'$Q^{2}$',fontsize=14)
    plt.xlabel(r'$f$',fontsize=14)
    plt.legend()
    plt.show()
    
gaussian_func_2dim_easy = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (0)**(2))))  \
                                        *np.exp(-(1./(2*(1 - (0)**(2))))*(x**(2) + y**(2) - 2*(0)*x*y))
                                        
gaussian_func_2dim_extra = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (0)**(2))))*x*y  \
                                        *np.exp(-(1./(2*(1 - (0)**(2))))*(x**(2) + y**(2) - 2*(0)*x*y))                                        

def two_pt_easy(th):
    """Should give f^(2)"""
    res = integrate.dblquad(gaussian_func_2dim_easy, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]

def eo_numerical(th):
    """Should give same as theory"""
    res = integrate.dblquad(gaussian_func_2dim_extra, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]



"""Compute readout error & SNR """
def compute_err_and_snr(N,P,H,d_in,th,spars=False):
    """Always {0,1}"""
    n_real = 50
    errors = np.zeros(n_real)
    len_test = P
    for j in range(n_real):
        stim = make_patterns(N,P)
        labels = np.zeros(P)
        for i in range(P):
            labels[i] = make_labels(0.5)
        
        patts_test = np.zeros((N,len_test))
        labels_test = labels
        ints = np.arange(P)
        ##CREATE TEST PATTERNS
        for n in range(len_test):#Pick test points - perturb ONE pattern randomly
            patt_typ = stim[:,n]
            patt_test = flip_patterns_cluster(patt_typ,d_in)
            d_in_check = compute_delta_out(patt_test,patt_typ)
            patts_test[:,n] = patt_test

        h,h_test = random_proj_generic_test(H,stim,patts_test,th,sparse=spars)
        #h,h_test = structured_proj_generic_test(H,stim,patts_test,th)
        
        o_spars = 0.5*(np.sign(h)+1)
        o = np.sign(h)
        o_test = np.sign(h_test)
        cod = compute_sparsity(o_spars[:,np.random.randint(P)])
        f = erf1(th)
        #f = cod #For structured weights, we take the empirical sparsity
        #print("coding",f)
        
        o_test_spars = 0.5*(np.sign(h_test)+1)

        o_in = o_spars - f
        o_test_in = o_test_spars - f
        
        w_hebb = np.matmul(o_in,labels) 
        
        stabs = []
        d_outs = []
        acts_typ = np.zeros((H,len_test))
        f_in = erf1(th) 
        #f_in = cod #Empirically obtain coding, since theoretically it is a bit more involved
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            stabs.append(stab)
            d_out = (1/(2*f_in*(1-f_in))) * compute_delta_out(o_test_in[:,m],o_in[:,ints[m]])
            d_outs.append(d_out)
            acts_typ[:,m] = o_in[:,ints[m]]
        
        d_out_mean = np.mean(d_outs)
        d_std = np.std(d_outs)
        #print("d_out_mean",d_out_mean)
        
        d_out_theory = erf_full(th,d_in,f_in)
        #print("d_out theory",d_out_theory)
        
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errors[j] = err
        
    err_mean = np.mean(errors)
    err_std = np.std(errors)
  
    numer_theory = (1 - d_out_theory)
    #th_in = th/(np.sqrt(1 - d_in))
    th_in = th
    denom_theory = P/H + (P/N) * excess_over_theory(th_in,f_in)**(2)
    #print("excess over before divided by f",(1/N)*excess_over_theory(th,f_in)**(2) * (f_in*(1-f_in))**(2))
  
    snr_theory = (numer_theory**(2))/denom_theory
    err_theory = erf1(np.sqrt(snr_theory))

    return err_mean, err_std, err_theory, cod,f, d_out_mean, d_out_theory

# N=100
# P=50
# H=2100
# ds=0.1
# th=0.6
# err = compute_err_and_snr(N,P,H,ds,th,spars=True)[0]
# print("error is",err)

# cod_theory = erf1(th/np.sqrt(0.07))
# print("cod theory",cod_theory)

run_readout_err = False
if run_readout_err:
    N=100
    P=50
    H=2100
    
    thress = np.linspace(0.0,0.5,10)
    #thress = [1.1,1.5,1.8]
    ds_list = [0.1,0.3,0.5] #Actually a list of p1=p2, but p3 is determined
    cods = np.zeros((len(thress)))
    f_empirical = np.zeros((len(thress)))
    #ds_list = np.linspace(0.1,1.0,10) #For sweep vs. \Delta \xi
    
    err_empirical = np.zeros((len(ds_list),len(thress)))
    err_theorys = np.zeros((len(ds_list),len(thress)))
    
    #f_empirical = np.zeros(len(thress))
    
    frac = 7/N
    
    for j,th in enumerate(thress):
        f_theory = erf1(th/np.sqrt(frac))
        cods[j] = f_theory
        print("coding theory",f_theory)
        
        for i,ds in enumerate(ds_list):
            
            err_mean, err_std, err_theory, cod,f, d_out, d_out_theory = compute_err_and_snr(N,P,H,ds,th,spars=True)
            
            err_empirical[i,j] = err_mean
            #print("d_out",d_out)
            
            ### Cluster size is p1*d1(ds) + p2*d2(dc) + p3*(deff)
            #err_theorys[i,j] = d_out_theory
            #print("d_out_theory",d_out_theory)
        
        f_empirical[j] = cod

    
    plt.figure()
    x_list = [1,2,3]
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    plt.title(r'Sparse, random, feed-forward weights - $K=7$')
    for i,ds in enumerate(ds_list):
        clr = next(colors)
        clr_theory = next(colors_ver)
        plt.plot(cods,err_empirical[i,:],'s',label=r'$\Delta \xi={}$'.format(ds),color=clr)
        #plt.plot(cods,err_theorys[i,:],'--',color=clr_theory)
    plt.xlabel(r'$f$',fontsize=14)
    plt.ylabel(r'$\epsilon$',fontsize=14)
    plt.legend()
    plt.show()
    
    # plt.title("Sparsity - structured feedforward (unimodal)")
    # plt.plot(f_empirical,thress,'s',color=clr)
    # plt.plot(cods,thress,'--',color=clr_theory)
    # plt.xlabel(r'$T$',fontsize=14)
    # plt.ylabel(r'$f$',fontsize=14)
    # plt.legend()
    # plt.show()
    


###PLOT CLUSTER SIZE
# N=100
# P=100
# H=2100
# ds = 0.1
# th = 3.0
# stim = make_patterns(N,P)
# patts_test = np.zeros((N,P))

# for i in range(stim.shape[0]):
#     patts_test[:,i] = flip_patterns_cluster(stim[:,i],ds)
    
# h,h_test = random_proj_generic_test(H,stim,patts_test,th)
# o = 0.5*(np.sign(h)+1)
# o_test = 0.5*(np.sign(h_test)+1)
# cod = compute_sparsity(o[:,np.random.randint(P)])

# d_emps_list = []
# for i in range(stim.shape[1]):
#     d_emp = compute_delta_out(o[:,i],o_test[:,i])
#     d_emps_list.append(d_emp)

# f = erf1(th)
# print("f is",f)
# d_emp_mean = np.mean(d_emps_list) / (2*f*(1-f))
# d_theory = erf_full(th,ds,f)
# # print("d_theory",d_theory)
# print("d_emp",d_emp_mean)
                                     

#Simulate sparse feed-forward network
#K=7
#N=100
#rat_in = K/N
#P=100
#stim = make_patterns(N,P)
#H=2000
#th=0.2
#h = random_proj_generic(H,stim,th,sparse=True)
#o = 0.5*(np.sign(h)+1)
#cod_emp = compute_sparsity(o[:,np.random.randint(P)])
#cod = erf1(th/(np.sqrt(rat_in)))
#print("cod_emp",cod_emp)
#print("cod",cod)
#
#pr_emp,pr_theory,fp = compute_pr_theory_sim(o-cod,th/np.sqrt(1),N)
#print("pr_emp",pr_emp)
#print("pr_theory",pr_theory)
#
#err_mean, err_std, err_theory, f = compute_err_and_snr(N,P,H,0.1,th,spars=True)
#print("theoretical error",err_theory)
#print("empirical error",err_mean)


    
#N=100
#P=200
#H=4000
#stim = make_patterns(N,P)
#thress = np.linspace(0.1,3.1,20)
#cods = np.zeros(len(thress))
#pr_emps = np.zeros(len(thress))
#pr_theorys = np.zeros(len(thress))
#fp_corrs = np.zeros(len(thress))
#for i,th in enumerate(thress):
#    h = random_proj_generic(H,stim,th)
#    o = 0.5*(np.sign(h)+1)
#    #f = compute_sparsity(o[:,np.random.randint(P)])
#    cod = erf1(th)
#    cods[i] = cod
#    print("sparsity is",f)
#    pr1,pr2,fp = compute_pr_theory_sim(o - cod,th,N)
#    print("pr is",pr1)
#    pr_emps[i] = pr1
#    pr_theorys[i] = pr2
#    fp_corrs[i] = fp/((cod*(1-cod))**(2))
#    print("fp_corrs",fp/((cod*(1-cod))**(2)))
#    
#plt.figure()
#plt.title(r'Re-scaled interference - very sparse ($N=100$,$\mathcal{R}=40$,$P=200$)',fontsize=12)
#plt.plot(cods,fp_corrs,'s-')
#plt.xlabel(r'$f$',fontsize=14)
#plt.ylabel(r'Re-scaled inteference')
#plt.show()
#    
#plt.figure()
#plt.title(r'Dimensionality - very sparse ($N=100$,$\mathcal{R}=40$,$P=200$)',fontsize=12)
#plt.plot(cods,pr_emps,'s')
#plt.plot(cods,pr_theorys,'--')
#plt.xlabel(r'$f$',fontsize=14)
#plt.ylabel(r'Dimensionality')
#plt.show()
#
#plt.figure()
#plt.title(r'Dimensionality - very sparse ($N=100$,$\mathcal{R}=40$,$P=200$)',fontsize=12)
#plt.plot(fp_corrs,pr_emps,'s')
#plt.plot(fp_corrs,pr_theorys,'--')
#plt.xlabel(r'$\frac{\langle{\mathcal{I}_{4}}\rangle}{f^{2}(1-f)^{2}}$',fontsize=14)
#plt.ylabel(r'Dimensionality')
#plt.show()


###Plot sparseness and dimensionality for two separate weight choices
run_dim_compare_sparse_dense = False
if run_dim_compare_sparse_dense:
    K=7
    N = 100
    P = 100
    stim = make_patterns(N,P)
    H = 2000
    ths = np.linspace(0.0,1.1,20)
    #hs = {}
    cods1 = np.zeros(len(ths))
    cods2 = np.zeros(len(ths))
    excess_over_one = np.zeros(len(ths))
    excess_over_two = np.zeros(len(ths))
    pr_dense = np.zeros(len(ths))
    pr_dense_theory = np.zeros(len(ths))
    pr_sparse = np.zeros(len(ths))
    pr_sparse_theory = np.zeros(len(ths))
    errors_dense = np.zeros(len(ths))
    errors_dense_theory = np.zeros(len(ths))
    errors_sparse = np.zeros(len(ths))
    errors_sparse_theory = np.zeros(len(ths))
    for i,th in enumerate(ths):
        h1 = random_proj_generic(H,stim,th,sparse=False)
        h2 = random_proj_generic(H,stim,th,sparse=True)
        o1 = 0.5*(np.sign(h1) + 1)
        o2 = 0.5*(np.sign(h2) + 1)
        cov1 = np.matmul(o1,o1.T)
        cov2 = np.matmul(o2,o2.T)
        cod1 = erf1(th)
        #print("cod1",cod1)
        cod2 = erf1(th/(np.sqrt(K/N)))
        print("cod2",cod2)
        cod2_emp = compute_sparsity(o2[:,np.random.randint(P)])
        print("cod2_emp",cod2_emp)
        #hs[i] = h
        cods1[i] = cod1
        cods2[i] = cod2
        
        err_mean, err_std, err_theory, f = compute_err_and_snr(N,P,H,0.1,th,spars=False)
        err_mean2, err_std2, err_theory2, f2 = compute_err_and_snr(N,P,H,0.1,th/np.sqrt(K/N),spars=True)
        
        errors_dense[i] = err_mean
        errors_dense_theory[i] = err_theory
        errors_sparse[i] = err_mean2
        print("error sparse",err_mean2)
        errors_sparse_theory[i] = err_theory2
        print("error sparse theory",err_theory2)
        
#        eo1 = (excess_over_theory(th,cod1))**(2)
#        eo2 = (excess_over_theory(th/(np.sqrt(K/N)),cod2))**(2)
#        excess_over_one[i] = eo1
#        #print("eo1",eo1)
#        excess_over_two[i] = eo2
#        print("eo2",eo2)
#        
#        pr1_emp, pr1_theory, fp1 = compute_pr_theory_sim(o1 - cod1,th,N)
#        pr2_emp, pr2_theory, fp2 = compute_pr_theory_sim(o2 - cod2,th/(np.sqrt(1)),N)
#        print("fp2",fp2/(cod2*(1-cod2))**(2))
#        print("pr_emp2",pr2_emp)
#        print("pr_theory2",pr2_theory)
#        
#        denom_calc = (1/(H*P)) + 1/P + 1/H + (1/N)*(fp2/((cod2*(1-cod2))**(2)))
#        #print("rescaled fp2",fp2/(cod2*(1-cod2))**(2))
#        #print("denom_calc",denom_calc)
#        #print("pr_calc2",1/denom_calc)
#        
#        pr_dense[i] = pr1_emp
#        pr_dense_theory[i] = pr1_theory
#        pr_sparse[i] = pr2_emp
#        pr_sparse_theory[i] = pr2_theory
      
    #colors = itertools.cycle(('green','blue','red','black'))
    #colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))  
    
    plt.figure()
    ax1 = plt.subplot(121)
    ax1.set_title('Dense connectivity',fontsize=14)
    ax1.plot(cods1,errors_dense,'s-',color='blue')
    #ax1.plot(cods1,errors_dense_theory,'--',label=r'Theory',color='lightblue')
    ax1.set_xlabel(r'$f$',fontsize=16)
    ax1.set_ylabel(r'$\epsilon$',fontsize=16)
    
    ax2 = plt.subplot(122)
    ax2.set_title('Sparse connectivity',fontsize=14)
    ax2.plot(cods2,errors_sparse,'s-',color='black')
    #ax2.plot(cods2,errors_sparse_theory,'--',label=r'Theory',color='grey')
    ax2.set_xlabel(r'$f$',fontsize=16)
    ax2.set_ylabel(r'$\epsilon$',fontsize=16)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# t_in=0.8
# th=t_in
# function_2dim_peak = lambda y,x: (1/(2*np.pi))*np.exp(-(1./2)*(x**(2) + y**(2)))

# def func_2dim_peak(p=0):
#     exp_ = lambda y,x: np.exp(p*x*y)
#     return exp_

# def func_3dim(x,y,p):
#     exp_ = np.exp(-(1./(2*(1-p**(2))))*(x**(2) + y**(2)) + p*x*y)
#     denom = (1/(2*np.pi))
#     return denom * exp_

# def func_3dim_g(x,y,p):
#     exp_ = x*y*np.exp(-(1./(2*(1-p**(2))))*(x**(2) + y**(2)) + p*x*y)
#     denom = (1/(2*np.pi))
#     return denom * exp_


# p_in = 0.2
# res = integrate.dblquad(func_3dim_g, t_in, np.inf, lambda x: th, lambda x: np.inf,args = (p_in,))
# print(res[0])

