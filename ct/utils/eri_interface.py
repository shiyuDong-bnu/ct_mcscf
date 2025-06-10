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


class SlicedERI:
    def __init__(self, my_orbital_space):
        self.bs_obs = my_orbital_space.bs_obs()
        self.bs_cabs = my_orbital_space.bs_cabs()

        self.mints = psi4.core.MintsHelper(self.bs_obs)

        self.coeff_gbs = my_orbital_space.Cp
        self.coeff_cbs = my_orbital_space.Cx

        self.ao_int = {}
        self.mo_int = {}

        self.gen_ao_int()
        self.gen_mo_int()

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
    def format_g_for_fock(self):
        n_obs=self.coeff_gbs.shape[1]
        n_cbs=self.coeff_cbs.shape[-1]
        obs=slice(0,n_obs)
        cbs=slice(n_obs,n_obs+n_cbs)
        n_total=n_obs+n_cbs
        g1=np.empty((n_total,n_obs,n_total,n_obs)) # eq8 (g^{\mu\lambda}_{\nu\kappa})
        ## gggg
        ## cggg
        ## ggcg
        ## cgcg
        g1[obs,obs,obs,obs]=self.mo_int["g_pqrs"]
        g1[cbs,obs,obs,obs]=np.moveaxis(self.mo_int["g_pqrx"],[0,1,2],[3,2,1])
        g1[obs,obs,cbs,obs]=np.moveaxis(self.mo_int["g_pqrx"],[0,2],[1,3])
        g1[cbs,obs,cbs,obs]=np.moveaxis(self.mo_int["g_pxqy"],[0,2],[1,3])
        ## gggg
        ## cggg
        ## gggc
        ## cggc
        g2=np.empty((n_total,n_obs,n_obs,n_total))
        g2[:,obs,obs,obs]=g1[:,obs,obs,obs]
        g2[obs,obs,obs,cbs]=self.mo_int["g_pqrx"]
        g2[cbs,obs,obs,cbs]=np.swapaxes(self.mo_int["g_pqxy"],0,2)
        return (g1,g2)
    def format_cbar1(self):
        ## g_sscs 
        n_obs=self.coeff_gbs.shape[1]
        n_cbs=self.coeff_cbs.shape[-1]
        s=slice(0,n_obs) ## correspondint to index pqrst
        c=slice(n_obs,n_obs+n_cbs) # corresponding to index x,y
        return np.moveaxis(self.mo_int["g_pqrx"],[0,2],[1,3])

        
