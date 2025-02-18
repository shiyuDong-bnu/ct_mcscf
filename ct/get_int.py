import numpy as np
import psi4

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
    g_pqrs = mints.ao_eri(bs_obs, bs_obs, bs_obs, bs_obs).to_array().swapaxes(1,2)  # to phys notation
    print(s)
    print(Cp.shape)
    print(g_pqrs.shape)
    g[s,s,s,s] = np.einsum("pP,qQ,rR,sS,pqrs->PQRS", Cp, Cp, Cp, Cp, g_pqrs, optimize=True)

    # <pq|xy> and <xy|pq> and <xq|py> and <py|xq>
    g_pqxy = mints.ao_eri(bs_obs, bs_cabs, bs_obs, bs_cabs).to_array().swapaxes(1,2)  # <pq|xy> = (px|qy)
    g[s,s,c,c] = np.einsum("pP,qQ,xX,yY,pqxy->PQXY", Cp, Cp, Cx, Cx, g_pqxy, optimize=True)
    g[c,c,s,s] = g[s,s,c,c].transpose((2,3,0,1))
    g[c,s,s,c] = g[s,s,c,c].transpose((2,1,0,3))
    g[s,c,c,s] = g[s,s,c,c].transpose((0,3,2,1))

    # <px|qy> and <xp|yq>
    g_pxqy = mints.ao_eri(bs_obs, bs_obs, bs_cabs, bs_cabs).to_array().swapaxes(1,2)
    g[s,c,s,c] = np.einsum("pP,xX,qQ,yY,pxqy->PXQY", Cp, Cx, Cp, Cx, g_pxqy, optimize=True)
    g[c,s,c,s] = g[s,c,s,c].transpose((1,0,3,2))

    # <pq|rx> and <qp|xr> and <rx|pq> and <xr|qp>
    g_pqrx = mints.ao_eri(bs_obs, bs_obs, bs_obs, bs_cabs).to_array().swapaxes(1,2)
    g[s,s,s,c] = np.einsum("pP,qQ,rR,xX,pqrx->PQRX", Cp, Cp, Cp, Cx, g_pqrx, optimize=True)
    g[s,s,c,s] = g[s,s,s,c].transpose((1,0,3,2))
    g[s,c,s,s] = g[s,s,s,c].transpose((2,3,0,1))
    g[c,s,s,s] = g[s,s,s,c].transpose((3,2,1,0))
    return g
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
def get_density(my_orbital_space):
    no=my_orbital_space.no
    nbf=my_orbital_space.nbf
    o=my_orbital_space.o   
    delta = np.identity(no)
    D1 = np.zeros((nbf, nbf))
    D1[o,o] = 2 * delta[o,o]
    D2 = np.zeros((nbf,nbf,nbf,nbf))
    D2[o,o,o,o] = 4 * np.einsum("ij,kl->ikjl", delta[o,o], delta[o,o])
    D2[o,o,o,o] -= 2 * np.einsum("il,kj->ikjl", delta[o,o], delta[o,o])
    return D1,D2
def  get_fock(my_orbital_space,h,D1,g):
    o=my_orbital_space.o
    # build fock
    f = np.copy(h)
    f += np.einsum("lk,mlnk->mn", D1[o,o], g[:,o,:,o])
    f -= 0.5 * np.einsum("lk,mlkn->mn", D1[o,o], g[:,o,o,:])
    return f
