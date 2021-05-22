#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selectivity model
"""

from random_expansion import *
from dimensionality_disentanglement import *
from sparseness_expansion import *
import random as random


def arrange_composite_form(stim,cont1,cont2):
    N = stim.shape[0]
    P = stim.shape[1]
    M = cont1.shape[0]
    K = cont1.shape[1]
    arr_stim = np.zeros((N,P))
    arr_cont1 = np.zeros((M,P*K)) 
    arr_cont2 = np.zeros((M,P*K*K))
    
    mat_big = np.zeros((N+2*M,P*K*K))

    for l in range(K):
        tile_cont1 = np.tile(np.reshape(cont1[:,l],(M,1)),P)
        arr_cont1[:,l*P:(l+1)*P] = tile_cont1
    
    for l1 in range(K):    
        tile_cont2 = np.tile(np.reshape(cont2[:,l1],(M,1)),P*K)
        arr_cont2[:,l1*(P*K):(l1+1)*(P*K)] = tile_cont2
    
    for p in range(P):
        arr_stim[:,p] = stim[:,p]
    
    arr_stim_in = np.tile(arr_stim,K**(2))
    arr_cont1_in = np.tile(arr_cont1,K)
    
    mat_big[:N,:] = arr_stim_in
    mat_big[N:N+M,:] = arr_cont1_in
    mat_big[N+M:,:] = arr_cont2
    
    return mat_big

def decompose_from_composite(stim_eff,N,M,P,K):
    """
    To decompose composite pattern matrix into stimuli and contexts
    """
    stim = np.zeros((N,P))
    cont1 = np.zeros((M,K))
    cont2 = np.zeros((M,K))
    for p in range(P):
        stim[:,p] = stim_eff[:N,p]
        for k in range(K):
            cont1[:,k] = stim_eff[N:N+M,k*P]
            for l in range(K):
                cont2[:,l] = stim_eff[N+M:N+2*M,l*(P*K)]
                
    return stim, cont1, cont2
                


def flip_pattern_cluster_comp(stim_comp,N,M,d_stim,d_cont):
    """
    Flip patterns that are in a composite form
    """
    stim_out = np.zeros(len(stim_comp))
    
    stim = stim_comp[:N]
    cont = stim_comp[N:N+M]
    cont2 = stim_comp[N+M:]
    
    stim_test = flip_patterns_cluster(stim,d_stim)
    cont_test = flip_patterns_cluster(cont,d_cont)
    cont2_test = flip_patterns_cluster(cont2,d_cont)
    
    stim_out[:N] = stim_test
    stim_out[N:N+M] = cont_test
    stim_out[N+M:] = cont2_test
    
    return stim_out


def generate_order_full_mixed_test(H,N,M,P,K,d_stim,d_cont,len_test=1.0):
    
    """
    Same as above, but needs to have K >=5. Set K=10 hence 5 test contexts
    """
    
    
    stim = make_patterns(N,P)
    cont1 = make_patterns(M,K)
    cont2 = make_patterns(M,K)

    stim_eff = arrange_composite_form(stim,cont1,cont2) #Make sure that all are arranged in a (PK^(2) x Nc) matrix
    print("shape stim_eff",stim_eff.shape)
    
    Peff = P*K*K
    test_len = int(len_test*Peff)
    ints_test = np.arange(test_len)
    
    stim_test_in = np.zeros((N+2*M,test_len))
    
    for i in range(test_len):
        ind_ = ints_test[i]
        stim_typ = stim_eff[:,ind_]
        stim_test = flip_pattern_cluster_comp(stim_typ,N,M,d_stim,d_cont)
        stim_test_in[:,i] = stim_test 
        d_check = compute_delta_out(stim_test,stim_typ)


    mat1 = np.random.normal(0,1/np.sqrt(3*N),(H,N))
    mat2 = np.random.normal(0,1/np.sqrt(3*M),(H,M))
    mat3 = np.random.normal(0,1/np.sqrt(3*M),(H,M))
    
    h = np.zeros((H,Peff))
    h_test = np.zeros((H,test_len))
    
    stim_test, cont1_test, cont2_test = decompose_from_composite(stim_test_in,N,M,P,K)
    
    count=0
    #For loop for training data
    for p in range(P):
        h_stim = np.matmul(mat1,stim[:,p])
        h_test_stim = np.matmul(mat1,stim_test[:,p])
        for k in range(K):
            #count+=1
            h_cont1 = np.matmul(mat2,cont1[:,k])
            h_test_cont1 = np.matmul(mat2,cont1_test[:,k])
            for l in range(K):
                count+=1
                h_cont2 = np.matmul(mat3,cont2[:,l])
                h_test_cont2 = np.matmul(mat3,cont2_test[:,l])
                #print("shape of things multiplied",mat3.shape,stim_test_in[N+M:,p].shape)
                h_in = h_stim + h_cont1 + h_cont2
                h_in_test = h_test_stim + h_test_cont1 + h_test_cont2
                h[:,count-1] = h_in
                h_test[:,count-1] = h_in_test
    
                
    return h, h_test


def generate_order_two_mixed_test(H,N,M,P,K,d_stim,d_cont,len_test=0.1):
    """Same as above, but with test data"""

    stim = make_patterns(N,P)
    cont1 = make_patterns(M,K)
    cont2 = make_patterns(M,K)

    stim_eff = arrange_composite_form(stim,cont1,cont2) #Make sure that all are arranged in a (PK^(2) x Nc) matrix
    print("shape stim_eff",stim_eff.shape)
    
    Peff = P*K*K
    test_len = int(Peff)
    ints_test = np.arange(test_len)
    
    stim_test_in = np.zeros((N+2*M,test_len))
    print("shape test data",stim_test_in.shape)
    
    for i in range(test_len):
        ind_ = ints_test[i]
        stim_typ = stim_eff[:,ind_]
        stim_test = flip_pattern_cluster_comp(stim_typ,N,M,d_stim,d_cont)
        stim_test_in[:,i] = stim_test 
        d_check = compute_delta_out(stim_test,stim_typ)


    mat1 = np.random.normal(0,1/np.sqrt(2*N),(int(H/2),N))
    mat2 = np.random.normal(0,1/np.sqrt(2*M),(int(H/2),M))
    mat3 = np.random.normal(0,1/np.sqrt(2*M),(int(H/2),M))
    
    h = np.zeros((H,Peff))
    h_test = np.zeros((H,test_len))
    
    stim_test, cont1_test, cont2_test = decompose_from_composite(stim_test_in,N,M,P,K)
    
    count=0
    for p in range(P):
        for k in range(K):
            for l in range(K):
                h_stim = np.matmul(mat1,stim[:,p])
                h_stim_test = np.matmul(mat1,stim_test[:,p])
                h_cont = np.matmul(mat2,cont1[:,k])
                h_cont_test = np.matmul(mat2,cont1_test[:,k])
                h_cont2 = np.matmul(mat3,cont2[:,l])
                h_cont2_test = np.matmul(mat3,cont2_test[:,l])
                
                count += 1
                
                h[:int(H/2),count-1] = h_stim + h_cont
                h_test[:int(H/2),count-1] = h_stim_test + h_cont_test
                
                h[int(H/2):,count-1] = h_cont + h_cont2
                h_test[int(H/2):,count-1] = h_cont_test + h_cont2_test
    
    return h, h_test

#N=100
#M=100
#P=50
#K=3
#H=3000
#h,h_test = generate_order_two_mixed_test(H,N,M,P,K,0.1,0.1)
#o = np.sign(h)
#print("rank of o",LA.matrix_rank(o))
#
#w,succ = perceptron_storage(o)

def generate_order_one_mixed_test(H,N,M,P,K,d_stim,d_cont):
    
    """
    All P columns as above
    
    First H/3: Receive inputs one
    Next H/3: Same, etc.
    """
    stim = make_patterns(N,P)
    cont1 = make_patterns(M,K)
    cont2 = make_patterns(M,K)
    
    stim_eff = arrange_composite_form(stim,cont1,cont2) #Make sure that all are arranged in a (PK^(2) x Nc) matrix
    
    Peff = P*K*K
    test_len = int(Peff)
    ints_test = np.arange(test_len)
    
    stim_test_in = np.zeros((N+2*M,test_len))
    print("shape test data",stim_test_in.shape)
    
    for i in range(test_len):
        ind_ = ints_test[i]
        stim_typ = stim_eff[:,ind_]
        stim_test = flip_pattern_cluster_comp(stim_typ,N,M,d_stim,d_cont)
        stim_test_in[:,i] = stim_test 
        d_check = compute_delta_out(stim_test,stim_typ)   
        
    stim_test, cont1_test, cont2_test = decompose_from_composite(stim_test_in,N,M,P,K)
    
    h_big = np.zeros((H,P*K*K))
    h_big_test = np.zeros((H,P*K*K))
    
    mat1 = np.random.normal(0,1/np.sqrt(N),(int(H/3),N))
    mat2 = np.random.normal(0,1/np.sqrt(M),(int(H/3),M))
    mat3 = np.random.normal(0,1/np.sqrt(M),(int(H/3),M))
    
    arr_stim = np.zeros((int(H/3),P)) #To be repeated K^(2) times
    arr_stim_test = np.zeros((int(H/3),P))
    arr_cont1 = np.zeros((int(H/3),P*K)) #To be repeated K times
    arr_cont1_test = np.zeros((int(H/3),P*K))
    arr_cont2 = np.zeros((int(H/3),P*K*K))
    arr_cont2_test = np.zeros((int(H/3),P*K*K))

    for l in range(K):
        h_cont1 = np.matmul(mat2,cont1[:,l])
        h_cont1_test = np.matmul(mat2,cont1_test[:,l])
        tile_cont1 = np.tile(np.reshape(h_cont1,(int(H/3),1)),P)
        tile_cont1_test = np.tile(np.reshape(h_cont1_test,(int(H/3),1)),P)
        arr_cont1[:,l*P:(l+1)*P] = tile_cont1
        arr_cont1_test[:,l*P:(l+1)*P] = tile_cont1_test
    
    for l1 in range(K):    
        h_cont2 = np.matmul(mat3,cont2[:,l1])
        h_cont2_test =  np.matmul(mat3,cont2_test[:,l1])
        #tile_cont2 = np.repeat(np.reshape(h_cont2,(int(H/3),1)),P*K,axis=1)
        tile_cont2 = np.tile(np.reshape(h_cont2,(int(H/3),1)),P*K)
        tile_cont2_test = np.tile(np.reshape(h_cont2_test,(int(H/3),1)),P*K)
        arr_cont2[:,l1*(P*K):(l1+1)*(P*K)] = tile_cont2
        arr_cont2_test[:,l1*(P*K):(l1+1)*(P*K)] = tile_cont2_test
    
    for p in range(P):
        h_stim = np.matmul(mat1,stim[:,p])
        h_stim_test = np.matmul(mat1,stim_test[:,p])
        arr_stim[:,p] = h_stim
        arr_stim_test[:,p] = h_stim_test
    
    arr_stim_in = np.tile(arr_stim,K**(2))
    arr_stim_in_test = np.tile(arr_stim_test,K**(2))
    arr_cont1_in = np.tile(arr_cont1,K)
    arr_cont1_in_test = np.tile(arr_cont1_test,K)
    h_big[:int(H/3),:] = arr_stim_in
    h_big[int(H/3):int(2*H/3),:] = arr_cont1_in
    h_big[int(2*H/3):,:] = arr_cont2
    h_big_test[:int(H/3),:] = arr_stim_in_test
    h_big_test[int(H/3):int(2*H/3),:] = arr_cont1_in_test
    h_big_test[int(2*H/3):,:] = arr_cont2_test
    
    return h_big, h_big_test

def compute_cluster_size_theory(ds,dp,th):
    cod = erf1(th)
    ds_eff_full = (1/3)*(ds + 2*dp)
    ds_eff1 = (1/2)*(ds + dp)
    
    ds1 = (1/3)*(erf_full(th,ds,cod) + 2*erf_full(th,dp,cod))
    ds2 = (1/3)*(2*erf_full(th,ds_eff1,cod) + erf_full(th,dp,cod))
    ds3 = erf_full(th,ds_eff_full,cod)
    
    return ds1, ds2, ds3
     


###Eigenvalue sectrum for different mixing degrees
# N=100
# M=N
# P=100
# K=2
# Peff = int(P*K*K)
# H=2100
# th=0.8
# stim = make_patterns(N,P)
# h3,h_test3 = generate_order_full_mixed_test(H,N,M,P,K,0.1,0.1)
# h2,h_test2 = generate_order_two_mixed_test(H,N,M,P,K,0.1,0.1)
# h1,h_test1 = generate_order_one_mixed_test(H,N,M,P,K,0.1,0.1)
# o1 = 0.5*(np.sign(h1 - th) + 1)
# o2 = 0.5*(np.sign(h2 - th) + 1)
# o3 = 0.5*(np.sign(h3 - th) + 1)  

# h_u = random_proj_generic(H, stim)
# o_u = 0.5*(np.sign(h_u-th)+1)

# cov1 = (1/Peff) * np.matmul(o1,o1.T)
# cov2 =  (1/Peff) * np.matmul(o2,o2.T)
# cov3 =  (1/Peff) * np.matmul(o3,o3.T)
# cov_u = (1/P) * np.matmul(o_u,o_u.T)

# cov_h_u = (1/H)*np.matmul(h_u.T,h_u)
# cov_h_1 = (1/H)*np.matmul(h1.T,h1)
# cov_h_3 = (1/H)*np.matmul(h3.T,h3)

# fig = plt.figure(figsize=(10,5))
# plt.title(r'$\mathbf{C} = \mathbf{h}^{T}\mathbf{h}$ - rep. similarity matrix?',fontsize=14)
# ax1 = fig.add_subplot(131)
# ax1.set_title(r'Unimodal')
# im1 = ax1.imshow(cov_h_u)
# ax2 = fig.add_subplot(132)
# ax2.set_title(r'$\mathcal{M}=1$')
# im2 = ax2.imshow(cov_h_1)
# ax3 = fig.add_subplot(133)
# ax3.set_title(r'$\mathcal{M}=3$')
# im3 = ax3.imshow(cov_h_3)

# plt.colorbar(im1)

# plt.tight_layout()
# plt.show()


# plt.figure()
# plt.title(r'Eigenvalue spectrum comparison vs. $\mathcal{M}$, $N_{m}=3$ - tail')
# plt.plot(LA.eigvals(cov_u),color='black',label=r'Unimodal')
# plt.plot(LA.eigvals(cov1),color='orange',label=r'$\mathcal{M}=1$')
# plt.plot(LA.eigvals(cov2),color='green',label=r'$\mathcal{M}=2$')
# plt.plot(LA.eigvals(cov3),color='red',label=r'$\mathcal{M}=3$')
# plt.xlabel(r'Ranked eigenvalue index',fontsize=14)
# plt.ylabel(r'Eigenvalue',fontsize=14)
# plt.xlim(0,4)
# plt.ylim(50,250)
# plt.legend()
# plt.show()



### Capacity
        

    
####PLOT OF DELTA M
run_delta_m = False
if run_delta_m:
    N=100
    M=100
    P=20
    K=4
    H=2100
    
    dp = 0.1
    
    thress = np.linspace(0,3.1,20)
    ds_list = [0.001]
    cods = np.zeros((len(thress)))
    
    delta_emps_one = np.zeros((len(ds_list),len(thress)))
    delta_theorys_one = np.zeros((len(ds_list),len(thress)))

    delta_emps_two = np.zeros((len(ds_list),len(thress)))
    delta_theorys_two = np.zeros((len(ds_list),len(thress)))
    
    delta_emps_three = np.zeros((len(ds_list),len(thress)))
    delta_theorys_three = np.zeros((len(ds_list),len(thress)))

    for j,th in enumerate(thress):
        cod = erf1(th)
        cods[j] = cod
        
        delta_mat = np.zeros((len(ds_list),3))
        
        
        for i, ds in enumerate(ds_list):
            h1,h1_test = generate_order_one_mixed_test(H,N,M,P,K,ds,dp)
            h2, h2_test = generate_order_two_mixed_test(H,N,M,P,K,ds,dp) 
            h3, h3_test = generate_order_full_mixed_test(H,N,M,P,K,ds,dp)
            o = 0.5*(np.sign(h1 - th) + 1) - cod
            o_test = 0.5*(np.sign(h1_test - th) + 1) - cod
            o2 = 0.5*(np.sign(h2 - th) + 1) - cod
            o2_test = 0.5*(np.sign(h2_test - th) + 1) - cod
            o3 = 0.5*(np.sign(h3 - th) + 1) - cod
            o3_test = 0.5*(np.sign(h3_test - th) + 1) - cod
            
            d1_list = []
            d2_list = []
            d3_list = []
            
            for p in range(o.shape[1]):
                d1 = compute_delta_out(o[:,p],o_test[:,p])
                d1_list.append(d1)
                d2 = compute_delta_out(o2[:,p],o2_test[:,p])
                d2_list.append(d2)
                d3 = compute_delta_out(o3[:,p],o3_test[:,p])
                d3_list.append(d3)
                
            delta_emps_one[i,j] = np.mean(d1_list)/ compute
            delta_emps_two[i,j] = np.mean(d2_list)/ (2*cod*(1-cod))
            delta_emps_three[i,j] = np.mean(d3_list)/ (2*cod*(1-cod))
            
            ds_eff_full = (1/3)*(ds + 2*dp)
            ds_eff1 = (1/2)*(ds + dp)
            
            ds1, ds2, ds3 = compute_cluster_size_theory(ds,dp,cod)
            
            #delta_theorys_one[i,j] = erf_full(th,ds_eff1,cod)  
            delta_theorys_one[i,j] = ds1
            
            #delta_theorys_two[i,j] = erf_full(th,ds_eff_full,cod)
            delta_theorys_two[i,j] = ds2
            
            delta_theorys_three[i,j] = ds3
        
    
    plt.figure()
    x_list = [1,2,3]
    colors = itertools.cycle(('blue','red','black'))
    colors_ver = itertools.cycle(('lightskyblue','lightcoral','grey'))
    plt.title(r'$\Delta \xi = 0$, $\Delta \phi=0.1$',fontsize=12)
    #for i,ds in enumerate(ds_list):
        #clr = next(colors)
        #plt.plot(x_list,delta_mat[i,:],'s-',color=clr,label=r'$\Delta \xi={}$'.format(ds))
    plt.plot(cods,delta_emps_one[0,:],'s',color='blue',label=r'$\mathcal{M}=1$')
    plt.plot(cods,delta_theorys_one[0,:],'--',color='lightblue',label=r'Theory')
    plt.plot(cods,delta_emps_two[0,:],'s',color='red',label=r'$\mathcal{M}=2$')
    plt.plot(cods,delta_theorys_two[0,:],'--',color='lightcoral',label=r'Theory')
    plt.plot(cods,delta_emps_three[0,:],'s',color='green',label=r'$\mathcal{M}=3$')
    plt.plot(cods,delta_theorys_three[0,:],'--',color='lightgreen',label=r'Theory')
    plt.xlabel(r'$f$',fontsize=14)
    plt.ylabel(r'$\Delta m$',fontsize=14)
    plt.legend()
    plt.show()


######BELOW IS DIMENSIONALITY + HEBBIAN LEARNING##########


def excess_over_theory_multimod(th,f,ind,N,M,Nm=3):
    """
    Args: 
        Nm: Number of modalities
        ind: Index of selectivity, either 1,2, or 3
    """
    exp_ = np.exp(-th**(2))
    core = (1/(2*np.pi)) * exp_
    delta = M/N
    intra = (1/N)*core*(1 + 1/delta)
    inter = (1/N)*core*(1 + 1/delta)
    excess_over = intra + inter
    
    return excess_over


def excess_over_across_all(th,N,M):
    """
    Excess over with peak at zero
    """
    f = erf1(th)
    delta = M/N
    stem = np.exp(-2*(th)**(2))/((2*np.pi)**(2))
    coeff = (1/(9*N)) * ((delta + 2)/(delta)) * stem
    out = coeff/((f*(1-f))**(2))
    
    return out

def prob_across_all(P,K):
    num = (1 - 1/P)*(1 - 1/K)**(2)
    return num

def excess_over_across_one(th,N,M):
    """
    Excess over centered at 1/3 - first
    """
    f = erf1(th)
    delta = M/N
    stem = np.exp(-2*(th)**(2))/((2*np.pi)**(2))
    coeff = (1/(9*N)) * ((delta + 1)/(delta)) * stem
    out = coeff/((f*(1-f))**(2))
    
    return out

def prob_across_one(P,K):
    num = 2*(1 - 1/P)*(1/K)*(1 - 1/K)
    return num

def excess_over_context(th,N,M):
    """
    Excess over centered at 1/3 - second
    """
    f = erf1(th)
    delta = M/N
    stem = np.exp(-2*(th)**(2))/((2*np.pi)**(2))
    coeff = (2/(9*M)) * stem
    out = coeff/((f*(1-f))**(2))
    
    return out

def prob_across_cont(P,K):
    num = (1/P)*(1 - 1/K)**(2)
    return num

def excess_over_unimod_stim(th,N):
    """
    Excess over centered at 2/3 - stim
    """
    f = erf1(th)
    stem = np.exp(-2*(th)**(2))/((2*np.pi)**(2))
    coeff = (1/(9*N)) * stem
    out = coeff/((f*(1-f))**(2))
    
    return out

def prob_unimod_stim(P,K):
    num = (1 - 1/P)*(1/K)**(2)
    return num

def excess_over_unimod_cont(th,M):
    """
    Excess over centered at 2/3 - cont
    """
    f = erf1(th)
    stem = np.exp(-2*(th)**(2))/((2*np.pi)**(2))
    coeff = (1/(9*M)) * stem
    out = coeff/((f*(1-f))**(2))
    
    return out

def prob_unimod_cont(P,K):
    num = (1/(P*K)) * (1 - 1/K)
    return num
    
gaussian_func_2dim_onethird = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/3)**(2))))  \
                                        *np.exp(-(1./(2*(1 - (1/3)**(2))))*(x**(2) + y**(2) - 2*(1/3)*x*y))
                                       
gaussian_func_2dim_twothird = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (2/3)**(2)))) \
                                        *np.exp(-(1./(2*(1 - (2/3)**(2))))*(x**(2) + y**(2) - 2*(2/3)*x*y))                                        
                                        
gaussian_func_2dim_onehalf = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/2)**(2)))) \
                                        *np.exp(-(1./(2*(1 - (1/2)**(2))))*(x**(2) + y**(2) - 2*(1/2)*x*y))                                       

gaussian_func_onethird_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/3)**(2)))) * x * y  \
                                        *np.exp(-(1./(2*(1 - (1/3)**(2))))*(x**(2) + y**(2) - 2*(1/3)*x*y))

gaussian_func_twothird_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (2/3)**(2)))) * x * y \
                                        *np.exp(-(1./(2*(1 - (2/3)**(2))))*(x**(2) + y**(2) - 2*(2/3)*x*y))

gaussian_func_onehalf_eo = lambda y,x: (1/(2*np.pi*np.sqrt(1 - (1/2)**(2)))) * x * y \
                                        *np.exp(-(1./(2*(1 - (1/2)**(2))))*(x**(2) + y**(2) - 2*(1/2)*x*y))  


def two_pt(th,pk=1/3):
    if pk == 1/3:
        res = integrate.dblquad(gaussian_func_2dim_onethird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 2/3:
        res = integrate.dblquad(gaussian_func_2dim_twothird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/2:
        res = integrate.dblquad(gaussian_func_2dim_onehalf, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 0: #To check if unimodal results recovered
        res = integrate.dblquad(gaussian_func_2dim_easy, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]

def eo_multimod(th,pk=1/3):
    if pk == 1/3:
        res = integrate.dblquad(gaussian_func_onethird_eo, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 2/3:
        res = integrate.dblquad(gaussian_func_twothird_eo, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/2:
        res = integrate.dblquad(gaussian_func_onehalf_eo, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 0:
         res = integrate.dblquad(gaussian_func_2dim_extra, th, np.inf, lambda x: th, lambda x: np.inf)   
    return res[0]

def squared_integral(th,pk=1/3):
    if pk == 1/3:
        res = integrate.dblquad(r_integral_onethird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 2/3:
        res = integrate.dblquad(r_integral_twothird, th, np.inf, lambda x: th, lambda x: np.inf)
    elif pk == 1/2:
        res = integrate.dblquad(r_integral_onehalf, th, np.inf, lambda x: th, lambda x: np.inf)
    return res[0]


def compute_excess_over_multimod(N,M,P,K,th):
    erf = erf1(th)
    p1 = prob_across_all(P,K)
    p2 = prob_across_one(P,K)
    p3 = prob_across_cont(P,K) 
    p4 = prob_unimod_stim(P,K)
    p5 = prob_unimod_cont(P,K)
    
    pk=1/3
    feff2 = two_pt(th,pk)
    q2 = erf*(1-erf)
    i4 = feff2
    denom2_onethird = (i4 - erf**(2))**(2)/(q2**(2))
    
    pk2=2/3
    feff3 = two_pt(th,pk2)
    i4_2 = feff3
    denom2_twothird = (i4_2 - erf**(2))**(2)/(q2**(2))
    
    denom2_eo_main = (1/(3*N))*excess_over_theory(th,erf)**(2)
    
    eo_acc_one = (2/(9*N))*(eo_multimod(th,1/3)**(2))/(q2**(2))
    eo_acc_cont = (2/(9*M))*(eo_multimod(th,1/3)**(2))/(q2**(2))
    eo_acc_stim = (1/(9*N))*(eo_multimod(th,2/3)**(2))/(q2**(2))
    eo_acc_cont = (1/(9*M))*(eo_multimod(th,2/3)**(2))/(q2**(2))
    
    denom2 = (p2)*(denom2_onethird + eo_acc_one) + p3*(denom2_onethird + eo_acc_cont) + p1*denom2_eo_main \
    + (p4)*(denom2_twothird + eo_acc_stim) + (p5)*(denom2_twothird + eo_acc_cont)
    
    return denom2


def hebbian_mixed_layer_interpolate(H,N,M,P,K,th,index=3,ds=0.1,dc=0.):
    """
    comp_num: True if want to compare "numerical theory" to theory
    """
    n_real = 20
    errors = np.zeros(n_real)
    Peff = P*K*K
    len_test = int(Peff)
    erf = erf1(th)
    for j in range(n_real):
        labels = np.zeros(int(Peff))
        for i in range(int(Peff)):
            labels[i] = make_labels(0.5)
        
        if index == 1:
            h,h_test = generate_order_one_mixed_test(H,N,M,P,K,ds,dc)
        elif index == 2:
            h,h_test = generate_order_two_mixed_test(H,N,M,P,K,ds,dc)
        elif index == 3:
            h,h_test = generate_order_full_mixed_test(H,N,M,P,K,ds,dc)
        print("after random projection")
        
        o_spars = 0.5*(np.sign(h-th)+1)
        o = np.sign(h-th)
        o_test = np.sign(h_test-th)
        f = compute_sparsity(o_spars[:,np.random.randint(Peff)])
        
        o_test_spars = 0.5*(np.sign(h_test-th)+1)
        
        o_in = o_spars - f
        o_test_in = o_test_spars - f
        
        w_hebb = np.matmul(o_in,labels) 
        stabs = []
        d_outs = []
        acts_typ = np.zeros((H,len_test))
        labels_test = labels
        for m in range(len_test):
            stab = labels_test[m]*np.dot(w_hebb,o_test_in[:,m])
            stabs.append(stab)
            d_out = (1/(2*erf*(1-erf))) * compute_delta_out(o_test_in[:,m],o_in[:,m])
            d_outs.append(d_out)
            #print("d_out is",d_out)
            acts_typ[:,m] = o_in[:,m]
          
            
        err = (len(np.where(np.asarray(stabs)<0)[0]))/(len(stabs))
        errors[j] = err
            
    err_mean = np.mean(errors)
    err_std = np.std(errors)
    
    d_out_mean = np.mean(d_outs)
    print("d_out is",d_out_mean)

    ds1,ds2,ds3 = compute_cluster_size_theory(ds,dc,th)

    if index == 1:
        d_theory = ds1
    elif index == 2:
        d_theory = ds2
    elif index == 3:
        d_theory = ds3
    
    d_theory_out = d_theory
    
    q_theory_in = compute_excess_over_multimod(N,M,P,K,th)

    diff = 1 - d_theory_out
    numer = diff**(2)

    denom2 = Peff/H + Peff * q_theory_in 
    
    snr = (numer)/(denom2)
    snr_in = snr
    err_theory = erf1(snr_in**(0.5))
    
    return snr, err_mean, err_std, err_theory,f, q_theory_in, o_in


run_dimensionality = False
if run_dimensionality:
    N=100
    M=100
    P=50
    K=2
    H=4000
    stim = make_patterns(N,P)
    stim_test = np.zeros((N,P))
    for p in range(stim.shape[1]):
        stim_test[:,p] = flip_patterns_cluster(stim[:,p],0.1)
    thress = np.linspace(0.1,3.2,20)
    pr_emps = np.zeros(len(thress))
    pr_theorys = np.zeros(len(thress))
    fp_corrs = np.zeros(len(thress))
    cods = np.zeros(len(thress))
    for i,th in enumerate(thress):
        #print("P is",P)
        #print("H is",H)
        snr, err_mean, err_std, err_theory, cod, intef, o = hebbian_mixed_layer_interpolate(H,N,M,P,K,th,index=3,ds=0.01,dc=0.01)
        denom_dim = (1/(H*P*K**(2))) + 1/H + 1/(P*K**(2)) + intef
        pr_calc = 1/(denom_dim)
        pr_theorys[i] = pr_calc
        cov_o = np.matmul(o,o.T)
        pr_emp = compute_pr_eigvals(cov_o)
        pr_emps[i] = pr_emp
        fp_corrs[i] = intef
        cods[i] = cod
        
#    plt.figure()
#    import matplotlib.ticker as ticker
#    ax = plt.subplot(121)
#    ax.set_title(r'Dimensionality',fontweight="bold",fontsize=16)
#    ax.plot(cods,pr_emps,'s',markersize=8,color='blue')
#    ax.plot(cods,pr_theorys,'--',color='lightblue')
#    start1, end1 = ax.get_xlim()
#    ax.set_ylabel(r'$\mathcal{D}$',fontsize=16)
#    #ax.set_xlabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{\langle q_{2} \rangle^{2}}$',fontsize=16)
#    diff = fp_corrs[3] - fp_corrs[4]
#    print("start,end,diff",start1,end1,diff)
    #ax.set_xticks(np.arange(start1, end1, 3*diff))
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    
    plt.title(r'Dimensionality, $\mathcal{M}=3$, $P=50$,$K=2$',fontsize=14)
    plt.plot(cods,pr_emps,'s',markersize=8,color='blue')
    plt.plot(cods,pr_theorys,'--',color='lightblue',label=r'Theory')
    plt.xlabel(r'$f$',fontsize=14)
    plt.ylabel(r'$\mathcal{D}$',fontsize=14)
    plt.legend()
    plt.show()
    
#    ax2 = plt.subplot(122)
#    ax2.set_title(r'Re-scaled interference',fontweight="bold",fontsize=16)
#    ax2.plot(cods,fp_corrs,'s-',markersize=8,color='blue')
#    start2, end2 = ax2.get_xlim()
#    diff2 = cods[1] - cods[2]
#    print("start,end,diff",start2,end2,diff2)
#    #ax.plot(fp_corrs,pr_theorys,'--',color='lightblue')
#    ax2.set_xlabel(r'$f$',fontsize=16)
#    ax2.set_ylabel(r'$\frac{\langle \mathcal{I}_{4} \rangle}{\langle q_{2} \rangle^{2}}$',fontsize=16)
    #ax2.set_xticks(np.arange(start2, end2, 3*diff2))
    #ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    #ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2E'))
    
    plt.tight_layout()
 
    plt.show()

###READOUT ERROR FOR DIFFERENT DEGREES OF MIXING
run_sweep_ratio = False
"""
3 sweeps:
    1. For different f
    2. For different Delta_in
    3. For different alpha
"""
if run_sweep_ratio:
    N = 100 #N larger for {+,-}
    M = 100
    #H=20000
    #P = 50
    K = 3
    #K_list = [1,3,6]
    thress = np.linspace(0.8,2.9,10)
    alphas = [0.1,0.5,2.0]
    #alphas = np.linspace(0.1,4.0,10)
    #alphas = [0.02,0.1,0.4]
    delta = 0.1
    #thress = [1,500] #HERE THRESS "=" R
    P_arr = np.linspace(0.2*N,4*N,10)
    P_list = [int(p) for p in P_arr]
    H_arr = np.linspace(1.0*N,20*N,10)
    #H_list = [int(p) for p in H_arr]
    H_list = [100,200,600,1000,2000,6000,10000,14000,20000]
    cods = np.zeros((len(thress)))
    #H_list = [200]
    #H_list = [20*N]
    err_means = np.zeros((len(H_list),len(alphas)))
    err_stds = np.zeros((len(H_list),len(alphas)))
    err_theorys = np.zeros((len(H_list),len(alphas)))
    snrs = np.zeros((len(H_list),len(alphas)))
    for j,h in enumerate(H_list):
        for i,a in enumerate(alphas):
            th=0.8
            #delta = d
            H = int(h)
            #H = 2100
            P=int(a*N)
            #print("P is",P)
            #print("H is",H)
            snr, err_mean, err_std, err_theory, cod, fp, o = hebbian_mixed_layer_interpolate(H,N,M,P,K,th,index=3,ds=0.01,dc=0.01)
            err_means[j,i] = err_mean
            print("empirical error",err_mean)
            err_theorys[j,i] = err_theory
            print("theoretical error",err_theory)
            err_stds[j,i] = err_std
            
            snrs[j,i] = snr
        #cods[j] = cod
    #np.savetxt("err_mean_largeN_R=10_modtheta_diffP.csv",err_means,delimiter=',')
    #np.savetxt("err_theory_largeN_R=10_modtheta_diffP.csv",err_theorys,delimiter=',')
    #np.savetxt("snrs_largeN_R=10_modtheta_diffP.csv",snrs,delimiter=',')
    
    #err_means = np.genfromtxt('err_mean_largeN_R=10_modtheta_diffP.csv',delimiter=',')
    #err_theorys = np.genfromtxt('err_theory_largeN_R=10_modtheta_diffP.csv',delimiter=',')
    colors = itertools.cycle(('green','blue','red','black'))
    colors_ver = itertools.cycle(('lightgreen','lightskyblue','lightcoral','grey'))
    plt.figure()
    plt.title(r'$\mathcal{M}=3$,$\Delta \xi = \Delta \phi=0.1$,$f=0.2$,$K=3$',fontsize=12)
    for i,ds in enumerate(alphas):
        clr = next(colors)
        clr_theory = next(colors_ver)
        plt.errorbar((1/N)*np.asarray(H_list),err_means[:,i],yerr=0.1*err_stds[:,i],color=clr,fmt='s',
                     capsize=5, markeredgewidth=2,label=r'$\beta={}$'.format(ds))
        plt.plot((1/N)*np.asarray(H_list),err_theorys[:,i],'--',color=clr_theory)
    plt.xlabel(r'$\mathcal{R}$',fontsize=14)
    plt.ylabel(r'Readout error',fontsize=14)
    plt.legend(fontsize=10)
    plt.show()



