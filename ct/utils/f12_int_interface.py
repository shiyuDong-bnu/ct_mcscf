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
        self.orbital_space = my_orbital_space
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
    def form_f12_moint(self):
        Cx=self.coeff_cbs
        Cp=self.coeff_gbs
        o=self.orbital_space.o
        v=self.orbital_space.v
        QF_xyij = self.ao_int["f12_cgcg"].swapaxes(1,2)
        QF_xaij = self.ao_int["f12_cggg"].swapaxes(1,2)
        QF_XYIJ=np.einsum("xX,yY,iI,jJ,xyij->XYIJ", Cx, Cx, Cp[:,o], Cp[:,o], QF_xyij, optimize=True)
        QF_XAIJ=np.einsum("xX,aA,iI,jJ,xaij->XAIJ", Cx, Cp[:,v], Cp[:,o], Cp[:,o], QF_xaij, optimize=True)
        self.mo_int["qf_xyij"] = QF_XYIJ
        self.mo_int["qf_xaij"] = QF_XAIJ
    def form_v_and_x_moint(self):
        """
        form the V and X mo integral
        the index is confusion now ,need to be fixed.
        """
        Cx=self.coeff_cbs
        Cp=self.coeff_gbs
        o=self.orbital_space.o
        v=self.orbital_space.v
        C_occ=Cp[:,o]

        rv_gggg=self.ao_int["f12g12_gggg"]
        r_ggga=self.ao_int["f12_gggc"]
        r_gggg=self.ao_int["f12_gggg"] 
        rr_gggg=self.ao_int["f12_squared_gggg"]  
        rv_gggg_phy=np.einsum("iajb->ijab",rv_gggg)
        r_ggga_phy=np.einsum("iajb->ijab",r_ggga)
        r_gggg_phy=np.einsum("iajb->ijab",r_gggg)
        rr_gggg_phy=np.einsum("iajb->ijab",rr_gggg)

        self.mo_int["rv_ijpq"]=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rv_gggg_phy,C_occ,C_occ,Cp,Cp,optimize=True)
        self.mo_int["r_ijpq"]=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_gggg_phy,C_occ,C_occ,Cp,Cp,optimize=True)
        self.mo_int["r_ijoa"]=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_ggga_phy,C_occ,C_occ,C_occ,Cx,optimize=True)
        self.mo_int["rr_ijkl"]=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rr_gggg_phy,C_occ,C_occ,C_occ,C_occ,optimize=True)
    def form_b_moint(self):
        Cx=self.coeff_cbs
        Cp=self.coeff_gbs
        o=self.orbital_space.o
        v=self.orbital_space.v
        n_occ=o.stop
        n_gbs=self.orbital_space.nbf
        n_cabs=self.orbital_space.ncabs
        n_ri=self.orbital_space.nri
        C_occ=Cp[:,o]
        d_com_ao=self.ao_int["double_commutator_gggg"]
        d_com_ao_phy=np.einsum("iajb->ijab",d_com_ao)
        self.mo_int["d_com_mo"]=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",d_com_ao_phy,C_occ,C_occ,
        C_occ,C_occ,optimize=True)

        rr_gggc_ao=self.ao_int["f12_squared_gggc"]
        rr_gggg_ao=self.ao_int["f12_squared_gggg"]
        rr_gggc_ao_phy=np.einsum("iajb->ijab",rr_gggc_ao)
        rr_gggg_ao_phy=np.einsum("iajb->ijab",rr_gggg_ao)

        rr_ooop_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rr_gggg_ao_phy,
                            C_occ,
                            C_occ,
                            C_occ,
                            Cp,optimize=True)
        rr_oooc_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rr_gggc_ao_phy,
                            C_occ,
                            C_occ,
                            C_occ,
                            Cx,optimize=True)
        self.mo_int["rr_ooori"]=np.concatenate((rr_ooop_mo,rr_oooc_mo),axis=-1)

        r_ggga=self.ao_int["f12_gggc"]
        r_gggg=self.ao_int["f12_gggg"]
        r_gaga=self.ao_int["f12_gcgc"]
        r_ggga_phy=np.einsum("iajb->ijab",r_ggga)
        r_gggg_phy=np.einsum("iajb->ijab",r_gggg)
        r_ggaa_phy=np.einsum("iajb->ijab",r_gaga)
        r_oocc_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_ggaa_phy,C_occ,C_occ,
                            Cx,Cx,optimize=True)
        r_oopc_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_ggga_phy,C_occ,C_occ,
                            Cp,Cx,optimize=True)
        r_oopq_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_gggg_phy,C_occ,C_occ,
                            Cp,Cp,optimize=True)
        r_oo_ri_ri_mo=np.zeros((n_occ,n_occ,n_ri,n_ri))
        r_oo_ri_ri_mo[:,:,:n_gbs,:n_gbs]=r_oopq_mo
        r_oo_ri_ri_mo[:,:,:n_gbs,n_gbs:]=r_oopc_mo
        r_oo_ri_ri_mo[:,:,n_gbs:,:n_gbs]=np.einsum("ijkl->jilk",r_oopc_mo)
        r_oo_ri_ri_mo[:,:,n_gbs:,n_gbs:]=r_oocc_mo
        self.mo_int["r_oo_ri_ri_mo"]=r_oo_ri_ri_mo

