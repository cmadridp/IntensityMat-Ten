#from polynomial import generate_new_basis_vector
from wavelet import WaveletNewBasis
import numpy as np

from tensor_training_stage import *


class generate_estimated_basis_vector:
    def __init__(self, dim, M):
        self.basis = WaveletNewBasis(M)
        self.dim = dim

    # basis_val[d][m] is the m-th basis evaluated at x[d]
    def compute_basis_val(self, P_x_basis, x):
        basis_val = [self.basis.compute_single_x_all_new_basis(P_x_basis[d], x[d]) for d in range(self.dim)]

        return basis_val

#generate_new_basis_vector(dim, P_x_basis,MM).compute_basis_val(X_train[0])
######################




class tensor_prediction:
    def __init__(self, tensor_shape, dim,M,X_train):


        self.compute_rank_one=compute_rank_one(dim)
        self.dim=dim 

        #compute projection basis matrix using iteration function
        self.cur_basis=iteration(  tensor_shape, dim,M,X_train).compute()
        self.generate_new_basis_vector = generate_estimated_basis_vector(dim, M)
        self.cur_shape=[]
        for dd in range(dim):
            self.cur_shape.append(len(self.cur_basis[dd]))
        
         
        self.A_predict=np.zeros(self.cur_shape, dtype=float)
        for i in range(len(X_train)):
            basis_val=self.generate_new_basis_vector.compute_basis_val(self.cur_basis, X_train[i])

            self.A_predict=self.A_predict+self.compute_rank_one.compute(basis_val)   
            
        self.A_predict=self.A_predict/len(X_train)

    
    def predict(self,x_test):
        
        basis_val=self.generate_new_basis_vector.compute_basis_val(self.cur_basis, x_test)
        ans= np.tensordot(basis_val[0],self.A_predict,axes=(0,0) )
        for dd in range(1,self.dim):
            ans= np.tensordot(basis_val[dd],ans,axes=(0,0) )
        return ans



