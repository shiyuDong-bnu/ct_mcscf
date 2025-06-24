import numpy as np
import psi4
from ct.utils.timer import timer_decorator
import time
import sys
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
        a_ind=mr_info.active_index
        o_ind=mr_info.inactive_docc_index
        ## 1rdm
        D1[o_ind,o_ind] = 2 * delta[o_ind,o_ind]
        D1[a_ind,a_ind]=mr_info.rdm1
        ## end copy from single reference

        ## 2rdm ,first index is occ
        D2[o_ind,:,:,:]=2*np.einsum("iq,rs->irqs",delta[o_ind,:],D1)
        D2[o_ind,:,:,:]-=np.einsum("is,rq->irqs",delta[o_ind,:],D1)
        ## first index is active
        ## second index is occ
        D2[a_ind,o_ind,:,:]=2*np.einsum("is,uq->uiqs",delta[o_ind,:],D1[a_ind,:])
        D2[a_ind,o_ind,:,:]-=np.einsum("iq,us->uiqs",delta[o_ind,:],D1[a_ind,:])
        ## second index is active
        D2[a_ind,a_ind,a_ind,a_ind]=mr_info.rdm2
    return D1,D2
def  get_fock(my_orbital_space,h,D1,g):
    s=my_orbital_space.s
    # build fock
    f_only_j =np.copy(h)+ np.einsum("lk,mlnk->mn", D1, g[0][:,s,:,s])
    k=0.5 * np.einsum("lk,mlkn->mn", D1, g[1][:,s,s,:])
    f_total=f_only_j-k
    return f_only_j,k,f_total
