import numpy as np
from ct.utils.timer import timer_decorator  

@timer_decorator
def get_hbar(my_orbital_space,V,X,B,g,G,f,h,rdm):
    """
    g class of sliced eri
    G  rational generator array
    D1 ,D2 density metrix
    f,h fock and core hamiltonian
    V term is of shape  [s,s,o,o] V^{pq}_{ij} -> V[p,q,i,j] # eq(23)
    X term is of shape  [o,o,o,o] X^{kl}_{ij} -> X[k,l,i,j] # eq(24)
    B term is of shape  [o,o,o,o] B^{kl}_{ij} -> B[k,l,i,j] # eq(25)
    """
    s=my_orbital_space.s
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v

    nbf=my_orbital_space.nbf
    # Eq. (28)
    #Dbar = 2 * np.einsum("pq,rs->prqs", D1, D1) - np.einsum("ps,rq->prqs", D1, D1) - D2
    D1=rdm.rdm1

   ## 1-body
    hbar=one_body(my_orbital_space,h,G,g,f,rdm)

    # 2-body
    Cbar2_p = two_body_decoposited(my_orbital_space,D1,g,G,f)
    # Eq. (20)
    Cbar2_p[:,:,o,o] += two_body_direct(my_orbital_space,V,X,B,G,f,h)
    # Eq. (16)
    gbar = g.mo_int["g_pqrs"]
    gbar[:,:,o,:] += 0.25 *Cbar2_p 
    gbar[:,:,:,o] += 0.25 *Cbar2_p.transpose((1,0,3,2)) 
    gbar[o,:,:,:] += 0.25 *Cbar2_p.transpose((2,3,0,1)) 
    gbar[:,o,:,:] += 0.25 *Cbar2_p.transpose((3,2,1,0)) 
    return hbar ,gbar
@timer_decorator
def one_body(my_orbital_space,h,G,g,f,rdm):
    debug=False
    s=my_orbital_space.s
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v
    nbf=my_orbital_space.nbf

    # 1-body
    Cbar1 = np.zeros((nbf,nbf))



    # Eq. (27), without 0.5
    ## temp is used for contraction of S ,which is deleted.
    temp = np.einsum('xaij,aa->xaij', G[c,v,o,o], f[v,v])
    temp -= np.einsum('xaij,ii->xaij', G[c,v,o,o], f[o,o])
    temp -= np.einsum('xaij,jj->xaij', G[c,v,o,o], f[o,o])

    # Eq. (18)
    g_ipxq,g_pixq=g.format_cbar1()
    d_bar_oooo=rdm.form_d_bar_oooo()
    d_bar_ooos=rdm.form_d_bar_ooos()
    Cbar1[v,s] += np.einsum("trij,trxq,xaij->aq",d_bar_oooo, g_ipxq[o,o,:,:], G[c,v,o,o],optimize='greedy')
    Cbar1[v,o] -= 2*np.einsum("tris,trxs,xaij->aj", d_bar_ooos, g_ipxq[o,o,:,:], G[c,v,o,o],optimize='greedy')
    Cbar1[v,o] += np.einsum("tris,trxs,xaji->aj", d_bar_ooos, g_ipxq[o,o,:,:], G[c,v,o,o],optimize='greedy')

    # Eq. (19)
    Cbar1[v,v] += 0.5 * np.einsum("klij,xaij,ybkl,xy->ab", d_bar_oooo, G[c,v,o,o], G[c,v,o,o], f[c,c], optimize="greedy")
    Cbar1[v,v] += 0.5 * np.einsum("klij,xaij,xbkl->ab", d_bar_oooo, temp, G[c,v,o,o], optimize="greedy")

    # Eq. (15)
    hbar = h[s,s] + 0.5 * Cbar1[s,s] + 0.5 * Cbar1.T
    return hbar
@timer_decorator
def two_body_direct(my_orbital_space,V,X,B,G,f,h):
    ## equation (20)
    s=my_orbital_space.s
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v

    nbf=my_orbital_space.nbf
    no=o.stop
    Cbar2 = np.zeros((nbf,nbf,no,no))
    Cbar2[s,v,o,o] += 4 * np.einsum("px,xbij->pbij", h[s,c], G[c,v,o,o])
    Cbar2[s,s,o,o] += 2 * V[s,s,o,o]
    Cbar2[s,o,o,o] -= 2 * np.einsum("klij,pk->plij", X[o,o,o,o], f[s,o])
    Cbar2[o,o,o,o] += B[o,o,o,o]
    return Cbar2
@timer_decorator
def two_body_decoposited(my_orbital_space,D1,g,G,f):
    debug=False
    s=my_orbital_space.s
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v

    nbf=my_orbital_space.nbf
    no=o.stop

    g_ipxq,g_pixq=g.format_cbar1()

    # Eq. (27), without 0.5
    temp = np.einsum("xaij,xy->yaij",G[c,v,o,o],f[c,c])
    temp += np.einsum('xaij,aa->xaij', G[c,v,o,o], f[v,v])
    temp -= np.einsum('xaij,ii->xaij', G[c,v,o,o], f[o,o])
    temp -= np.einsum('xaij,jj->xaij', G[c,v,o,o], f[o,o])

    # Eq. (21)
    Cbar2_p=np.zeros((nbf,nbf,no,nbf))
    Cbar2_p[v,s,o,s] += 4 * np.einsum("ti,trxs,xaij->arjs", D1[o,o], g_ipxq[o,:,:,:], G[c,v,o,o],optimize='greedy')


    Cbar2_p[v,s,o,s] -= 2 * np.einsum("ti,trxs,xaji->arjs", D1[o,o], g_ipxq[o,:,:,:], G[c,v,o,o],optimize='greedy')
    Cbar2_p[v,s,o,s] -= 2 * np.einsum("ti,rtxs,xaij->arjs", D1[o,o], g_pixq[:,o,:,:], G[c,v,o,o],optimize='greedy')
    


    Cbar2_p[s,v,o,o] += 4 * np.einsum("tu,ptxu,xaij->paij", D1[o,o], g_pixq[:,o,:,o], G[c,v,o,o],optimize='greedy')
    Cbar2_p[s,v,o,o] -= 2 * np.einsum("tu,tpxu,xaij->paij", D1[o,o],  g_ipxq[o,:,:,o], G[c,v,o,o],optimize='greedy')
    

    Cbar2_p[s,v,o,s] -= 2 * np.einsum("tj,ptxs,xaij->pais", D1[o,o], g_pixq[:,o,:,:], G[c,v,o,o],optimize='greedy')
    # Eq. (22)
    Cbar2_pp=np.zeros((no,nbf,nbf,nbf))
    Cbar2_pp[o,v,v,o] += 2 * np.einsum("xaij,xbkl,ki->labj", temp, G[c,v,o,o], D1[o,o],optimize="greedy")

    Cbar2_pp[o,v,v,o] -= np.einsum("xaij,xbkl,kj->labi", temp, G[c,v,o,o], D1[o,o],optimize="greedy")

    Cbar2_pp[o,v,v,o] -= np.einsum("xaij,xbkl,li->kabj", temp, G[c,v,o,o], D1[o,o],optimize="greedy")

    Cbar2_pp[o,v,o,v] -= np.einsum("xaij,xbkl,lj->kaib", temp, G[c,v,o,o], D1[o,o],optimize="greedy")

    # using permutation symmetry
    Cbar2_p[:,:,o,:] += Cbar2_pp.transpose((2,3,0,1))
    return Cbar2_p
def three_body(my_orbital_space,g,G,f):
    s=my_orbital_space.s
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v
    # Eq. (26)
    U = np.einsum("prxs,xbij->prsijb", g[s,s,c,s], G[c,v,o,o])

    # Eq. (27), without 0.5
    S = np.einsum("xaij,ybkl,xy->klbija", G[c,v,o,o], G[c,v,o,o], f[c,c], optimize=True)
    temp = np.einsum('xaij,aa->xaij', G[c,v,o,o], f[v,v])
    temp -= np.einsum('xaij,ii->xaij', G[c,v,o,o], f[o,o])
    temp -= np.einsum('xaij,jj->xaij', G[c,v,o,o], f[o,o])
    S += np.einsum("xaij,xbkl->klbija", temp, G[c,v,o,o])
    return U,S
