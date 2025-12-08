"""
This module is used to calculate two elctron eri 
which is a refactor of get_int.py
The aim is to get avoid of calculate eri(nri,nri,nri,nri)
the contraction do not need that large tensor
so we keep the eri be a sliced version
"""

import time
import sys
import psi4
import numpy as np
import torch


class SlicedERI:
    def __init__(self, my_orbital_space,int_wfn=None):
        self.bs_obs = my_orbital_space.bs_obs()
        self.bs_cabs = my_orbital_space.bs_cabs()
        self.n_occ=my_orbital_space.o.stop

        self.mints = psi4.core.MintsHelper(self.bs_obs)

        self.coeff_gbs = my_orbital_space.Cp
        self.coeff_cbs = my_orbital_space.Cx

        self.ao_int = {}
        self.mo_int = {}
        if int_wfn!=None:
            if int_wfn.variables()['SYDONG_DF']==1.0:
                self.load_df_mo_int()
            else:
                self.load_ao_int(int_wfn)
                self.gen_mo_int()
        else:
            self.gen_ao_int()
            self.gen_mo_int()
    def load_df_mo_int(self):
        print("loading eri integral from dfint_wfn")
        tensor_model = torch.jit.load("eri_tensors.pt")
        mo_pqrs = list(tensor_model.parameters())[0]

        mo_ijxy = list(tensor_model.parameters())[1]
        mo_ixjy = list(tensor_model.parameters())[2]
        mo_ipxq = list(tensor_model.parameters())[3]
        mo_pixq = list(tensor_model.parameters())[4]
        C_mo_pq = list(tensor_model.parameters())[5]
        T_ind_pq = list(tensor_model.parameters())[6]

        self.mo_int["g_pqrs"]=np.array(mo_pqrs)
        self.mo_int["g_pqxy"]=np.array(mo_ijxy)
        self.mo_int["g_pxqy"]=np.array(mo_ixjy)
        #self.mo_int["g_iqrx"]=np.array(mo_pqrx)
        self.mo_int["g_ipxq"]=np.array(mo_ipxq)
        self.mo_int["g_pixq"]=np.array(mo_pixq)
        self.mo_int["C_mo_pq"]=np.array(C_mo_pq)
        self.mo_int["T_ind_pq"]=np.array(T_ind_pq)
    def load_ao_int(self,int_wfn):
        print("loading eri integrals from int_wfn")
        result=int_wfn.variables()
        n_gbs=self.bs_obs.nbf()
        n_cabs=self.bs_cabs.nbf()

        self.ao_int["g_pqrs"]=result["g_pqrs".upper()].np.reshape(n_gbs,n_gbs,n_gbs,n_gbs)
        self.ao_int["g_pqxy"]=result["g_pqxy".upper()].np.reshape(n_gbs,n_cabs,n_gbs,n_cabs)
        self.ao_int["g_pxqy"] =result["g_pxqy".upper()].np.reshape(n_gbs,n_gbs,n_cabs,n_cabs)
        self.ao_int["g_pqrx"]=result["g_pqrx".upper()].np.reshape(n_gbs,n_gbs,n_gbs,n_cabs)
    def gen_ao_int(self):
        """
        ao integral is in chemist's notation
        """
        mints = self.mints
        bs_obs = self.bs_obs
        bs_cabs = self.bs_cabs
        begin = time.time()
        self.ao_int["g_pqrs"] = mints.ao_eri(
            bs_obs, bs_obs, bs_obs, bs_obs
        ).to_array()  # to phys notation
        psi4.core.set_global_option("SCREENING", "NONE")
        self.ao_int["g_pqxy"] = mints.ao_eri(
            bs_obs, bs_cabs, bs_obs, bs_cabs
        ).to_array()  # <pq|xy> = (px|qy)
        self.ao_int["g_pxqy"] = mints.ao_eri(
            bs_obs, bs_obs, bs_cabs, bs_cabs
        ).to_array()
        self.ao_int["g_pqrx"] = mints.ao_eri(bs_obs, bs_obs, bs_obs, bs_cabs).to_array()
        end = time.time()
        print(
            f"{ sys._getframe(  ).f_code.co_name} time to do integrals in ", end - begin
        )

    def gen_mo_int(self):
        """
        mo integral in in physicts's notation

        eightfold symmetry can be used to save memory
        """
        c_gbs = self.coeff_gbs
        c_cbs = self.coeff_cbs
        self.mo_int["g_pqrs"] = np.einsum(
            "pP,qQ,rR,sS,prqs->PQRS", ## in eninsum the ao index 1,2 is swaped!!
            c_gbs,
            c_gbs,
            c_gbs,
            c_gbs,
            self.ao_int["g_pqrs"],
            optimize="greedy",
        )
        self.mo_int["g_pqxy"] = np.einsum(
            "pP,qQ,xX,yY,pxqy->PQXY", ## in eninsum the ao index 1,2 is swaped!!
            c_gbs,
            c_gbs,
            c_cbs,
            c_cbs,
            self.ao_int["g_pqxy"],
            optimize="greedy",
        )
        self.mo_int["g_pxqy"] = np.einsum(
            "pP,xX,qQ,yY,pqxy->PXQY", ## in eninsum the ao index 1,2 is swaped!!
            c_gbs,
            c_cbs,
            c_gbs,
            c_cbs,
            self.ao_int["g_pxqy"],
            optimize="greedy",
        )
        self.mo_int["g_pqrx"] = np.einsum(
            "pP,qQ,rR,xX,prqx->PQRX", ## in eninsum the ao index 1,2 is swaped!!
            c_gbs,
            c_gbs,
            c_gbs,
            c_cbs,
            self.ao_int["g_pqrx"],
            optimize="greedy",
        )
        ## slice the mo int to fitting into the same patter of df mo int 
        n_occ=self.n_occ
        occ=slice(0,n_occ)
        self.mo_int["g_ipxq"]=np.moveaxis(self.mo_int["g_pqrx"],[0,1,2,3],[1,0,3,2])[occ,:,:,:]
        self.mo_int["g_pixq"]=np.moveaxis(self.mo_int["g_pqrx"],[0,1,2,3],[1,0,3,2])[:,occ,:,:]

        del self.ao_int
    def format_g_for_fock(self):
        n_obs=self.coeff_gbs.shape[1]
        n_cbs=self.coeff_cbs.shape[-1]
        n_occ=self.n_occ
        obs=slice(0,n_obs)
        cbs=slice(n_obs,n_obs+n_cbs)
        occ=slice(0,n_occ)
        n_total=n_obs+n_cbs
        g1=np.empty((n_total,n_occ,n_total,n_occ)) # eq8 (g^{\mu\lambda}_{\nu\kappa})
        ## gggg
        ## cggg
        ## ggcg
        ## cgcg
        # df g_pqrs

        C_mo_pq=self.mo_int["C_mo_pq"]
        T_ind_pq=self.mo_int["T_ind_pq"]
        temp_j=np.einsum("Apq,Aij->pqij",C_mo_pq,T_ind_pq[:,occ,occ])
        temp_j=np.moveaxis(temp_j,[0,1,2,3],[0,2,1,3]) ## to phy notation

        temp_k=np.einsum("Api,Ajs->pijs",C_mo_pq[:,:,occ],T_ind_pq[:,occ,:])
        temp_k=np.moveaxis(temp_k,[0,1,2,3],[0,2,1,3]) ## to phy notation

       # g1[obs,occ,obs,occ]=self.mo_int["g_pqrs"][:,occ,:,occ]
        g1[obs,occ,obs,occ]=temp_j
        g1[cbs,occ,obs,occ]=np.moveaxis(self.mo_int["g_pixq"],[0,1,2,3],[2,3,0,1])[:,occ,:,occ]
        g1[obs,occ,cbs,occ]=self.mo_int["g_pixq"][:,occ,:,occ]
        g1[cbs,occ,cbs,occ]=np.moveaxis(self.mo_int["g_pxqy"],[0,2],[1,3])[:,occ,:,occ]
        ## gggg
        ## cggg
        ## gggc
        ## cggc
        g2=np.empty((n_total,n_occ,n_occ,n_total))
        #g2[obs,occ,occ,obs]=self.mo_int["g_pqrs"][:,occ,occ,:]
        g2[obs,occ,occ,obs]=temp_k
        g2[cbs,occ,occ,obs]=np.moveaxis(self.mo_int["g_ipxq"],[0,1,2,3],[2,3,0,1])[:,occ,occ,:]
        g2[obs,occ,occ,cbs]=np.moveaxis(self.mo_int["g_ipxq"],[0,1,2,3],[1,0,3,2])[:,occ,occ,:]
        g2[cbs,occ,occ,cbs]=np.swapaxes(self.mo_int["g_pqxy"],0,2)[:,occ,occ,:]
        return (g1,g2)
    def format_cbar1(self):
        ## g_sscs 
        n_obs=self.coeff_gbs.shape[1]
        n_cbs=self.coeff_cbs.shape[-1]
        return  self.mo_int["g_ipxq"],self.mo_int["g_pixq"]

        
