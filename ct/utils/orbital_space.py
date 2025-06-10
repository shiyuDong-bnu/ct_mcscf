class OrbitalSpace():
    def __init__(self,wfn,obs,ribs,cabs):
        self.wfn=wfn
        self.obs=obs
        self.ribs=ribs
        self.cabs=cabs
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
    def no(self):
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
        occ C
        """
        return slice(0,self.no)
    @property
    def v(self):
        """
        vir in gbs B
        """
        return slice(self.no,self.nbf)
    @property
    def a(self):
        """
        all vir  A+B
        """
        return slice(self.no,self.nri)
    @property
    def s(self):
        """
        gbs   D
        """        
        return slice(0, self.nbf)
    @property
    def c(self):
        """
        cabs  A
        """    
        return slice(self.nbf, self.nbf + self.ncabs)
