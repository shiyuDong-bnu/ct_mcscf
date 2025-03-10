import numpy as np
class DIIS:
    def __init__(self,max_n=10):
        self.max_n=max_n
        self.fock=[]
        self.error=[]
    def update(self,fock,error):
        if len(self.fock)>=self.max_n:
            self.fock.pop(0)
            self.error.pop(0)
        self.fock.append(fock)
        self.error.append(error)
    def extropolate(self):
        n_dim=len(self.fock)+1
        self.right_hand=np.zeros(n_dim)
        self.right_hand[-1]=1
        self.left_hand=np.ones((n_dim,n_dim))
        self.left_hand[-1,-1]=0
        for i in range(n_dim-1):
            for j in range(i,n_dim-1):
                inner_prod=np.trace(self.error[i].T@self.error[j])
                self.left_hand[i,j]=self.left_hand[j,i]=inner_prod
        solution=np.linalg.solve(self.left_hand,self.right_hand)
        new_fock=np.zeros_like(self.fock[0])
        for i in range(n_dim-1):
            new_fock+=solution[i]*self.fock[i]
        return new_fock
    def calc_error(self,fock,density_matrix,overlap_matrix,S_sqrt):
        x=fock@density_matrix@overlap_matrix-overlap_matrix@density_matrix@fock
        return S_sqrt@x@S_sqrt
