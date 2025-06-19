class OrbitalSpace():
    def __init__(self,wfn,obs,ribs,cabs,mr_info=None):
        self.wfn=wfn
        self.obs=obs
        self.ribs=ribs
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
        return self.cabs.C().to_array()
    def bs_obs(self):
        return self.obs.basisset()
    def bs_cabs(self):
        return self.cabs.basisset()
    @property
    def nalpha(self):
        return self.wfn.nalpha()
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
