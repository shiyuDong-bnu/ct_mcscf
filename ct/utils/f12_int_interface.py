"""
This module is used to generate all f12 related integral 
once, and store them in a class.
Two motivations to do this :
1. avoid redundance calculation
2. easy to interface with c++ parallel generated integral.
"""    
import psi4
import numpy as np
class F12_INT:
    def __init__(self, my_orbital_space,gamma):
        self.bs_obs = my_orbital_space.bs_obs()
        self.bs_cabs = my_orbital_space.bs_cabs()

        self.mints = psi4.core.MintsHelper(self.bs_obs)

        self.coeff_gbs = my_orbital_space.Cp
        self.coeff_cbs = my_orbital_space.Cx

        self.ao_int = {}
        self.mo_int = {}
        self.cgtg = self.mints.f12_cgtg(gamma)
        self.gen_ao_int()
    def gen_ao_int(self):
        """
        ao integral is in chemist's notation
        f12_cgcg means integral type is ao_f12 , the basis type is cabs obs cabs obs
        """
        #  Fist f12 integral is used in get_f12 function.
        f12_cgcg = self.mints.ao_f12(self.cgtg, self.bs_cabs, self.bs_obs, self.bs_cabs, self.bs_obs).to_array()
        f12_cggg = self.mints.ao_f12(self.cgtg, self.bs_cabs, self.bs_obs, self.bs_obs, self.bs_obs).to_array()
        self.ao_int["f12_cgcg"]=f12_cgcg
        self.ao_int["f12_cggg"]=f12_cggg