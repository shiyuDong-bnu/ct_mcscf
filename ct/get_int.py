import numpy as np
import psi4
from ct.utils.timer import timer_decorator
import time
import sys
@timer_decorator
def get_eri_ri_ri_int(my_orbital_space):
    bs_obs=my_orbital_space.bs_obs()
    bs_cabs=my_orbital_space.bs_cabs()
    nri=my_orbital_space.nri
    Cp=my_orbital_space.Cp
    Cx=my_orbital_space.Cx
    s=my_orbital_space.s
    c=my_orbital_space.c
    mints = psi4.core.MintsHelper(bs_obs)
    psi4.core.set_global_option("SCREENING", "NONE")
    # for simplicity, let's get all 2e integrals
    g = np.zeros((nri, nri, nri, nri))
    
    # <pq|rs>
    begin=time.time()
    g_pqrs = mints.ao_eri(bs_obs, bs_obs, bs_obs, bs_obs).to_array()  # to phys notation
    g_pqxy = mints.ao_eri(bs_obs, bs_cabs, bs_obs, bs_cabs).to_array() # <pq|xy> = (px|qy)
    g_pxqy = mints.ao_eri(bs_obs, bs_obs, bs_cabs, bs_cabs).to_array()
    g_pqrx = mints.ao_eri(bs_obs, bs_obs, bs_obs, bs_cabs).to_array()
    end=time.time()
    print(f"{ sys._getframe(  ).f_code.co_name} time to do integrals in ",end-begin)

    begin=time.time()
    g[s,s,s,s] = np.einsum("pP,qQ,rR,sS,prqs->PQRS", Cp, Cp, Cp, Cp, g_pqrs, optimize="greedy")
    end=time.time()
    print(f"{ sys._getframe(  ).f_code.co_name} time to do integrals transformation ",end-begin)
    # <pq|xy> and <xy|pq> and <xq|py> and <py|xq>
    
    g[s,s,c,c] = np.einsum("pP,qQ,xX,yY,pxqy->PQXY", Cp, Cp, Cx, Cx, g_pqxy, optimize="greedy")
    #path_info = np.einsum_path("pP,qQ,xX,yY,pxqy->PQXY", Cp, Cp, Cx, Cx, g_pqxy, optimize="greedy")
    #print("Path info for Eq. (18):", path_info[0])
    #print("Path info for Eq. (18):", path_info[1])
    begin=time.time()
    g[c,c,s,s] = g[s,s,c,c].transpose((2,3,0,1))
    g[c,s,s,c] = g[s,s,c,c].transpose((2,1,0,3))
    g[s,c,c,s] = g[s,s,c,c].transpose((0,3,2,1))
    end=time.time()
    print(f"{ sys._getframe(  ).f_code.co_name} time to do array transpose ",end-begin)
    # <px|qy> and <xp|yq>
   
    g[s,c,s,c] = np.einsum("pP,xX,qQ,yY,pqxy->PXQY", Cp, Cx, Cp, Cx, g_pxqy, optimize="greedy")
    g[c,s,c,s] = g[s,c,s,c].transpose((1,0,3,2))

    # <pq|rx> and <qp|xr> and <rx|pq> and <xr|qp>
    
    g[s,s,s,c] = np.einsum("pP,qQ,rR,xX,prqx->PQRX", Cp, Cp, Cp, Cx, g_pqrx, optimize="greedy")
    g[s,s,c,s] = g[s,s,s,c].transpose((1,0,3,2))
    g[s,c,s,s] = g[s,s,s,c].transpose((2,3,0,1))
    g[c,s,s,s] = g[s,s,s,c].transpose((3,2,1,0))
    return g
@timer_decorator
def get_hcore_int(my_orbital_space):
    nri=my_orbital_space.nri
    bs_obs=my_orbital_space.bs_obs()
    bs_cabs=my_orbital_space.bs_cabs()
    Cp=my_orbital_space.Cp
    Cx=my_orbital_space.Cx
    s=my_orbital_space.s
    c=my_orbital_space.c

    mints = psi4.core.MintsHelper(bs_obs)
    # build 1e intergals
    h = np.zeros((nri, nri))

    h_pq = mints.ao_kinetic(bs_obs, bs_obs).to_array() + mints.ao_potential(bs_obs, bs_obs).to_array()
    h[s,s] = np.einsum("mp,nq,mn->pq", Cp, Cp, h_pq, optimize=True)

    h_px = mints.ao_kinetic(bs_obs, bs_cabs).to_array() + mints.ao_potential(bs_obs, bs_cabs).to_array()
    h[s,c] = np.einsum("mp,nq,mn->pq", Cp, Cx, h_px, optimize=True)
    h[c,s] = h[s,c].transpose()

    h_xy = mints.ao_kinetic(bs_cabs, bs_cabs).to_array() + mints.ao_potential(bs_cabs, bs_cabs).to_array()
    h[c,c] = np.einsum("mp,nq,mn->pq", Cx, Cx, h_xy, optimize=True)
    return h

# Density
@timer_decorator
def get_density(my_orbital_space,mr_info=None):
    nbf=my_orbital_space.nbf
    delta = np.identity(nbf)
    D1 = np.zeros((nbf, nbf))
    D2 = np.zeros((nbf,nbf,nbf,nbf))
    if mr_info is None:
        ## single reference closed shell
        o=my_orbital_space.o   
        D1[o,o] = 2 * delta[o,o]

        D2[o,o,o,o] = 4 * np.einsum("ij,kl->ikjl", delta[o,o], delta[o,o])
        D2[o,o,o,o] -= 2 * np.einsum("il,kj->ikjl", delta[o,o], delta[o,o])
    if mr_info is  not None:
        ## multireference
        a_ind=mr_info['active_index']
        o_ind=mr_info['occupied_index']
        ## 1rdm
        D1[o_ind,o_ind] = 2 * delta[o_ind,o_ind]
        D1[a_ind,a_ind]=mr_info['RDM1']
        ## end copy from single reference

        ## 2rdm ,first index is occ
        D2[o_ind,:,:,:]=2*np.einsum("iq,rs->irqs",delta[o_ind,:],D1)
        D2[o_ind,:,:,:]-=np.einsum("is,rq->irqs",delta[o_ind,:],D1)
        ## first index is active
        ## second index is occ
        D2[a_ind,o_ind,:,:]=2*np.einsum("is,uq->uiqs",delta[o_ind,:],D1[a_ind,:])
        D2[a_ind,o_ind,:,:]-=np.einsum("iq,us->uiqs",delta[o_ind,:],D1[a_ind,:])
        ## second index is active
        D2[a_ind,a_ind,a_ind,a_ind]=mr_info['RDM2']
    return D1,D2
def  get_fock(my_orbital_space,h,D1,g):
    s=my_orbital_space.s
    # build fock
    f = np.copy(h)
    f += np.einsum("lk,mlnk->mn", D1, g[0][:,s,:,s])
    f -= 0.5 * np.einsum("lk,mlkn->mn", D1, g[1][:,s,s,:])
    return f
