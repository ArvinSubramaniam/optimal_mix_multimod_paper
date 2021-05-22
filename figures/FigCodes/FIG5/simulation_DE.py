import numpy as np

import argparse, subprocess
# Simulation codes

class hetero_forward2(object):
    
    def __init__(self,N,M,P,K,Nc,p1,p2,p3):
        
        self.dim_mod0 = N  # neurons in first modality
        self.dim_mod1 = M  # neruons in second modality
        self.num_mod0 = P  # independent sample numbers of first modality
        self.num_mod1 = K  # independent sample numbers of second modality
        self.dim_cort = Nc # Neurons on cortical layer
        self.p1 = p1 # fraction of type 1
        self.p2 = p2 # fraction of type 2
        self.p3 = p3 # fraction of type 3
    
    # ------- generate binary random input ------#
    def generate_input(self):
        
        N = self.dim_mod0
        M = self.dim_mod1
        P = self.num_mod0
        K = self.num_mod1
        
        self.data_0 = np.sign(np.random.randn(P, N))
        self.data_1 = np.sign(np.random.randn(K, M))
        
    # ------- generate normalized random connection matrix ------#
    def random_connection(self):
        
        N = self.dim_mod0
        M = self.dim_mod1
        Nc = self.dim_cort
        
        Nc1 = np.int(Nc*self.p1)
        Nc2 = np.int(Nc*self.p2)
        Nc3 = np.int(Nc-Nc1-Nc2)
        
        self.J0_1 = np.random.normal(0,1/np.sqrt(N),size = (N,Nc1))  # Jsigma_1
        self.J1_1 = np.random.normal(0,1/np.sqrt(M),size = (M,Nc2))  # Jeta_1
        self.J0_2 = np.random.normal(0,1/np.sqrt(2*N),size = (N,Nc3))  # Jsigma_2
        self.J1_2 = np.random.normal(0,1/np.sqrt(2*M),size = (M,Nc3))  # Jeta_2
        
        
    # -----fix the sparsity in the cortical layer-----#
    def fix_sparsity(self, a, f=0.5):
        
        '''
        Input: matrix_h
        Return: matrix_m, T
        '''

        v = a.copy()
        threshold = np.sort(v.flatten())[int((1 - f) * v.size)]

        exite = v >= threshold
        inhibit = v < threshold

        v[exite] = 1
        v[inhibit] = 0

        return v,threshold
    
    def feed(self,f1=0.5, f2=0.5, f3=0.5, initial_data = False, return_m = True, initial_J = True):
        
        '''
        if return_m == True: return h,m,T   else: return h 
        '''
        
        P = self.num_mod0
        K = self.num_mod1
        Nc = self.dim_cort
        
        Nc1 = np.int(Nc*self.p1)
        Nc2 = np.int(Nc*self.p2)
        Nc3 = np.int(Nc-Nc1-Nc2)
        
        if initial_data: self.generate_input()
        if initial_J: self.random_connection()
        
        
        mix_layer_h = np.zeros((P * K, Nc))
        count = 0
        
        for i in range(P):
            
            for j in range(K):
                
                part1 = np.matmul(self.data_0[i],self.J0_1) 
                part2 = np.matmul(self.data_1[j],self.J1_1)
                part3 = np.matmul(self.data_0[i],self.J0_2) + np.matmul(self.data_1[j],self.J1_2)
                
                mix_layer_h[count,:] = np.concatenate((part1,part2,part3))
                
                count = count + 1
        
        if return_m == True:
            
            after_nonlinear = np.zeros((P*K,Nc))
            
            if Nc1 > 0 : after_nonlinear[:,0:Nc1],T1 = self.fix_sparsity(mix_layer_h[:,0:Nc1],f=f1)
            if Nc2 > 0 : after_nonlinear[:,Nc1:(Nc2+Nc1)],T2 = self.fix_sparsity(mix_layer_h[:,Nc1:(Nc2+Nc1)],f=f2)
            if Nc3 > 0 : after_nonlinear[:,(Nc2+Nc1):Nc],T3 = self.fix_sparsity(mix_layer_h[:,(Nc2+Nc1):Nc],f=f3)
            
            
            return mix_layer_h, after_nonlinear

        else:

            return mix_layer_h


        
def cal_delta_m(m,m_center,f=0.5):
    
    Nr,Nc = m.shape
    
    res = np.sum(np.abs(m-m_center))
    
    return res/(2*Nc*Nr*f*(1-f))

def flip_matrix(data,noise_parameter):
    flip_vector = np.sign(np.random.rand(data.shape[0],data.shape[1])-noise_parameter/2.)
    return data*flip_vector

def hebbian_weight(matrix_C,labels,f=0.5): 
    
    # m - f, make mean zero
    norm_C = matrix_C.copy() - f
    
    W = np.matmul(labels,norm_C)
    
    return W

def hebbian_readout(matrix_C,weight,f=0.5):
    
    # m - f, make mean zero
    norm_C = matrix_C.copy() - f
    
    y = np.sign(np.matmul(norm_C,weight))
    
    return y

def readout_error(y,labels):
    
    return((y!=labels).sum()/float(y.size))

def dimension(m):
    C = np.cov(m,rowvar=False)
    return np.trace(C)**2 / np.sum(C**2)

if __name__ == '__main__':
    # Primary parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-P','--number_P', type=int, default=20, help='Sample number in the first modality')
    parser.add_argument('-K','--number_K', type=int, default=20, help='Sample number in the second modality')
    parser.add_argument('-dx','--delta_xi', type=float, default=0.3, help='noise level in the first modality')
    parser.add_argument('-dp','--delta_phi', type=float, default=0.3, help='noise level in the second modality')
    parser.add_argument('-o','--outname', type=str, required=True, help='output name')
    
    args = parser.parse_args()  

    # Some shared parameters
    N=100
    M=100
    Nc=2100
    f=0.05
    repeat_num1 = 150
    repeat_num2 = 100
    p2=0

    # Read parameters from the command line
    P=args.number_P
    K=args.number_K
    delta_xi=args.delta_xi
    delta_phi=args.delta_phi
    print('P={},K={},dx={},dp={}'.format(P,K,delta_xi,delta_phi))

    simu_curve=[]
    for p1 in np.linspace(0.,1.,11):
        print('p1 = {}'.format(p1)) 
        p3 = 1.- p1
        temp_error_record = np.zeros(repeat_num1*repeat_num2)
        count = 0
        for repeat1 in range(repeat_num1):   
            model2 = hetero_forward2(N,M,P,K,Nc,p1,p2,p3)
            model2.generate_input()
            labels = np.sign(np.random.randn(P*K))
            
            h,m = model2.feed(f,f,f)
            m_center = m.copy()
            W_m = hebbian_weight(m,labels,f=f)
            center_0 = model2.data_0.copy()
            center_1 = model2.data_1.copy()

            for repeat2 in range(repeat_num2):

                model2.data_0 = flip_matrix(center_0,delta_xi)
                model2.data_1 = flip_matrix(center_1,delta_phi)

                h,m_noise = model2.feed(f,f,f,initial_J=False)
                y_m_noise = hebbian_readout(m_noise,W_m,f=f)

                temp_error_record[count] = readout_error(y_m_noise,labels)
                count = count + 1
        
        simu_curve.append(np.mean(temp_error_record))
    np.savetxt(args.outname, simu_curve, fmt='%.5f', header='P={},K={},dx={},dp={}'.format(P,K,delta_xi,delta_phi))
    print("Finished!")







