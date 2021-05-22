import numpy as np
from scipy.stats import norm
from scipy import integrate


#=========== Related Functions ============

def signal_coeff(f,T,delta_sigma,delta_eta):
    
    delta_mix = (delta_sigma + delta_eta) / 2.
    G1 = function_G(delta_sigma,f,T)
    G2 = function_G(delta_eta,f,T)
    G3 = function_G(delta_mix,f,T)
    
    return G1**2, G2**2, G3**2, 2*G1*G2, 2*G1*G3, 2*G2*G3, -2*G1, -2*G2, -2*G3

def part3_coeff(Nc,N,M,P,K,f,T,delta_sigma,delta_eta):
    
    delta_mix = (delta_sigma + delta_eta) / 2.
    G1 = function_G(delta_sigma,f,T)
    G2 = function_G(delta_eta,f,T)
    G3 = function_G(delta_mix,f,T)
    
    s1 = s_delta(delta_sigma)
    s2 = s_delta(delta_eta)
    
    I1 = function_I(T, delta_sigma)
    I2 = function_I(T, delta_eta)
    
    C0 = (1/Nc) * (P*K-1)* (f**2) * (1-f)**2 
    A = (1/Nc) * (2*f-1)**2 * (K-1) * f * (1-f) * (1-G1)
    B = (1/Nc) * (2*f-1)**2 * (P-1) * f * (1-f) * (1-G2)
    
    C = (1/Nc) * (2*f-1)**2 * (K-1) * s1 * I1 + (1/Nc) * (2*f-1)**2 * (P-1) * s2 * I2 - (1/Nc) * (2*f-1)**2 * f**2 * (P+K-2)
    
    return C0, A, B, C


def part4_coeff(Nc,N,M,P,K,f,T,delta_sigma,delta_eta):
    
    delta_mix = (delta_sigma + delta_eta) / 2.
    G1 = function_G(delta_sigma,f,T)
    G2 = function_G(delta_eta,f,T)
    G3 = function_G(delta_mix,f,T)
    
    s1 = s_delta(delta_sigma)
    s2 = s_delta(delta_eta)
    
    I1 = function_I(T, delta_sigma)
    I2 = function_I(T, delta_eta)
    
    g1 = function_g(T, delta_sigma)
    g2 = function_g(T, delta_eta)
    
    
    D = (K-1) * (f**2) * (1-f)**2 * (1-G1)**2 + (P*K-K) * (1/N) * (np.exp(-2*T**2)/(2*np.pi)**2) 
    E = (P-1) * (f**2) * (1-f)**2 * (1-G2)**2 + (P*K-P) * (1/M) * (np.exp(-2*T**2)/(2*np.pi)**2) 
    
    F1 = (K-1) * ( (s1*I1 - f**2)**2 + s1**2 * g1**2 * (1/(4*M)) )
    F2 = (P-1) * ( (s2*I2 - f**2)**2 + s2**2 * g2**2 * (1/(4*N)) )
    F3 = (P*K-P-K+1) * ( 1/(4*M) + 1/(4*N) ) * (np.exp(-2*T**2)/(2*np.pi)**2)
    
    R1 = (K-1) * (s1*I1 - f**2) * (1-G1) * f * (1-f)
    R2 = (P-1) * s2 * g2 * (1/(2*N)) * (np.exp(-T**2)/(2*np.pi)) 
    R3 = (P*K - P - K  + 1) * (1/(2*N)) * (np.exp(-2*T**2)/(2*np.pi)**2)
    
    S1 = (P-1) * (s2*I2 - f**2) * (1-G2) * f * (1-f)
    S2 = (K-1) * s1 * g1 * (1/(2*M)) * (np.exp(-T**2)/(2*np.pi)) 
    S3 = (P*K - P - K  + 1) * (1/(2*M)) * (np.exp(-2*T**2)/(2*np.pi)**2)
    
    return D, E, F1+F2+F3, R1+R2+R3, S1+S2+S3


def minus_SNR_p(p):
    
    p1, p2, p3 = p
    
    signal  = f**2 * (1-f)**2 * ( sig_coeff[0]*(p1**2) + sig_coeff[1]*(p2**2) + sig_coeff[2]*(p3**2) + 
                                sig_coeff[3]*p1*p2 + sig_coeff[4]*p1*p3 + sig_coeff[5]*p2*p3 
                                + sig_coeff[6]*p1 + sig_coeff[7]*p2 + sig_coeff[8]*p3 + 1)
    
    part3 = C0 + A*p1 + B*p2 + C*p3
    
    part4 = D*p1*p1 + E*p2*p2 + F*p3*p3 + 2*R*p1*p3 + 2*S*p2*p3
    
    return -signal/(part3+part4)

def C_p(p):
    
    p1, p2, p3 = p
    
    part3 = C0 + A*p1 + B*p2 + C*p3
    
    return part3

def Q2_p(p):
    
    p1, p2, p3 = p
    
    part4 = D*p1*p1 + E*p2*p2 + F*p3*p3 + 2*R*p1*p3 + 2*S*p2*p3
    
    return part4
















