import numpy as np
class OrbitalSpace():
    def __init__(self,wfn,obs,cabs,mr_info=None):
        self.wfn=wfn
        self.obs=obs
        self.cabs=cabs
        self.mr_info=None
        if mr_info !=None:
            print("MR Dimension INFORMATION ,CHECK!!!")
            print(mr_info)
            self.mr_info=mr_info
    @property
    def Cp(self):
        return self.obs.C().to_array()
    @property
    def Cx(self):
        np.save("ctcx.npy",self.cabs.C().to_array())
        return self.cabs.C().to_array()
    def bs_obs(self):
        return self.obs.basisset()
    def bs_cabs(self):
        return self.cabs.basisset()
    @property
    def nalpha(self):
        return self.wfn.nalpha()
    @property
    def nfrzc(self):
        return self.wfn.nfrzc()
    @property
    def nbf(self):
        return self.obs.dim().sum()
    @property
    def ncabs(self):
        return self.cabs.dim().sum()
    @property
    def nri(self):
        return self.ncabs+self.nbf
    @property
    def n_active_hole(self):
        """
        active hole :  restricted_docc + active
        """
        if self.mr_info is None:
            return self.nalpha-self.nfrzc 
        else:
            return self.mr_info.n_restricted_docc + self.mr_info.n_active
    @property
    def n_all_hole(self):
        """
        all hole: frozen_docc + restricted_docc + active
        """
        if self.mr_info is None:
            return  self.nalpha
        else:
            return self.mr_info.n_restricted_docc + self.mr_info.n_active +self.mr_info.n_frozen
    @property
    def o(self):
        """
        occ C {i,j,k,l,...}
        """
        if self.mr_info is not None:
            return self.mr_info.o
        return slice(0,self.nalpha)
    @property
    def v(self):
        """
        vir in gbs B {a,b,c,d,...}
        """
        if self.mr_info is not None:
            return self.mr_info.v
        return slice(self.o.stop,self.nbf)
    @property
    def a(self):
        """
        all vir  A+B {alhpa,beta,gamma,...}
        """
        return slice(self.o.stop,self.nri)
    @property
    def s(self):
        """
        gbs   D {p,q,r,s,...}
        """        
        return slice(0, self.nbf)
    @property
    def c(self):
        """
        cabs  A {x,y,z,...}
        """    
        return slice(self.nbf, self.nbf + self.ncabs)
