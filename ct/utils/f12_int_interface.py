"""
This module is used to generate all f12 related integral 
once, and store them in a class.
Two motivations to do this :
1. avoid redundance calculation
2. easy to interface with c++ parallel generated integral.
"""    
import time
import sys
import psi4
import numpy as np
class F12_INT:
    def __init__(self, my_orbital_space,gamma,int_wfn=None):
        self.bs_obs = my_orbital_space.bs_obs()
        self.bs_cabs = my_orbital_space.bs_cabs()

        self.mints = psi4.core.MintsHelper(self.bs_obs)

        self.coeff_gbs = my_orbital_space.Cp
        self.coeff_cbs = my_orbital_space.Cx

        self.ao_int = {}
        self.mo_int = {}
        self.cgtg = self.mints.f12_cgtg(gamma)
        if int_wfn!=None:
            self.load_f12_ao_int(int_wfn)
        else:
            self.gen_f12_ao_int()
    def load_f12_ao_int(self,int_wfn):
            print("loading f12 integrals from int_wfn")
            result=int_wfn.variables()
            n_gbs=self.bs_obs.nbf()
            n_cabs=self.bs_cabs.nbf()

            
            self.ao_int["f12_cgcg"]=result["F12_CGCG"].np.reshape(n_cabs,n_gbs,n_cabs,n_gbs)
            self.ao_int["f12_cggg"]=result ["F12_CGGG"].np.reshape(n_cabs,n_gbs,n_gbs,n_gbs)
            self.ao_int["f12_gggg"]=result ["F12_GGGG"].np.reshape(n_gbs,n_gbs,n_gbs,n_gbs)

  
            self.ao_int["f12g12_gggg"]=result ["F12G12_GGGG"].np.reshape(n_gbs,n_gbs,n_gbs,n_gbs)


            self.ao_int["f12_squared_gggg"]=result ["F12_SQUARED_GGGG"].np.reshape(n_gbs,n_gbs,n_gbs,n_gbs)
            self.ao_int["f12_squared_gggc"]=result ["F12_SQUARED_GGGC"].np.reshape(n_gbs,n_gbs,n_gbs,n_cabs)

       
            self.ao_int["double_commutator_gggg"]=result ["F12_DOUBLE_COMMUTATOR_GGGG"].np.reshape(n_gbs,n_gbs,n_gbs,n_gbs)
            
            self.ao_int["f12_gcgc"]=np.moveaxis(self.ao_int["f12_cgcg"],[0,1,2,3],[1,0,3,2])
            self.ao_int["f12_gggc"]=np.moveaxis(self.ao_int["f12_cggg"],[0,1,2,3],[3,2,1,0])        
    def gen_f12_ao_int(self):
        """
        ao integral is in chemist's notation
        f12_cgcg means integral type is ao_f12 , the basis type is cabs obs cabs obs
        """
        psi4.core.set_global_option("SCREENING", "NONE")
        begin = time.time()
        #  First f12 integral is used in get_f12 function.
        f12_cgcg = self.mints.ao_f12(self.cgtg, self.bs_cabs, self.bs_obs, self.bs_cabs, self.bs_obs).to_array()
        f12_cggg = self.mints.ao_f12(self.cgtg, self.bs_cabs, self.bs_obs, self.bs_obs, self.bs_obs).to_array()
        self.ao_int["f12_cgcg"]=f12_cgcg
        self.ao_int["f12_cggg"]=f12_cggg

        ## Second those integral is used in gen_V function
        f12g12_gggg=self.mints.ao_f12g12(self.cgtg,self.bs_obs,self.bs_obs,self.bs_obs,self.bs_obs)
        f12_gggg=self.mints.ao_f12(self.cgtg,self.bs_obs,self.bs_obs,self.bs_obs,self.bs_obs)
        f12_squared_gggg=self.mints.ao_f12_squared(self.cgtg,self.bs_obs,self.bs_obs,self.bs_obs,self.bs_obs)
        
        self.ao_int["f12g12_gggg"]=f12g12_gggg
        self.ao_int["f12_gggc"]=np.moveaxis(f12_cggg,[0,1,2,3],[3,2,1,0])
        self.ao_int["f12_gggg"]=f12_gggg
        self.ao_int["f12_squared_gggg"]=f12_squared_gggg

        ## Third those integral is used in gen_V function
        double_commutator_gggg=self.mints.ao_f12_double_commutator(self.cgtg,self.bs_obs,
                         self.bs_obs,self.bs_obs,self.bs_obs)
        f12_squared_gggc=self.mints.ao_f12_squared(self.cgtg,self.bs_obs,self.bs_obs,self.bs_obs,self.bs_cabs)        
        self.ao_int["double_commutator_gggg"]=double_commutator_gggg
        self.ao_int["f12_squared_gggc"]=f12_squared_gggc
        self.ao_int["f12_gcgc"]=np.moveaxis(f12_cgcg,[0,1,2,3],[1,0,3,2])

        end = time.time()
        print(
            f"{ sys._getframe(  ).f_code.co_name} time to do integrals in ", end - begin
        )
        ## 3 f12          cgcg cggg gggg 
        ## 1  f12g12       gggg 
        ## 2  f12 squared  gggg gggc
        ## 1 double_commutator gggg 
