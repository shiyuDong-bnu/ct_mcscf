import numpy as np
from ct.utils.timer import timer_decorator  

@timer_decorator
def get_hbar(my_orbital_space,V_rational,X_rational,B_rational,D1,D2,g,G,f,h):
    s=my_orbital_space.s
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v

    nbf=my_orbital_space.nbf
    V = np.zeros((nbf, nbf, nbf, nbf))
    V[s,s,o,o] = V_rational
    # Eq. (24)
    X = np.zeros((nbf, nbf, nbf, nbf))
    X[o,o,o,o] = X_rational
    # Eq. (25)
    B = np.zeros((nbf, nbf, nbf, nbf))
    B[o,o,o,o] =B_rational
    # Eq. (28)
    Dbar = 2 * np.einsum("pq,rs->prqs", D1, D1) - np.einsum("ps,rq->prqs", D1, D1) - D2

   ## 1-body
    hbar=one_body(my_orbital_space,h,Dbar,G,g,f)

    # 2-body
    Cbar2 = two_body_decoposited(my_orbital_space,D1,g,G,f)
    # Eq. (20)
    Cbar2 += two_body_direct(my_orbital_space,V,X,B,G,f,h)

    # Eq. (16)
    gbar = 0.25 * (np.copy(Cbar2) + Cbar2.transpose((1,0,3,2)))
    gbar += 0.25 * (Cbar2.transpose((2,3,0,1)) + Cbar2.transpose((3,2,1,0)))
    gbar += g.mo_int["g_pqrs"]
    return hbar ,gbar
@timer_decorator
def one_body(my_orbital_space,h,Dbar,G,g,f):
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
    g_sscs=g.format_cbar1()
    Cbar1[v,s] += np.einsum("trij,trxq,xaij->aq",Dbar[s,s,o,o], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:
        path_info = np.einsum_path("trij,trxq,xaij->aq",Dbar[s,s,o,o], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (18):", path_info[0])
        print("Path info for Eq. (18):", path_info[1])
    Cbar1[v,o] -= 2*np.einsum("tris,trxs,xaij->aj", Dbar[s,s,o,s], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:
        path_info = np.einsum_path("tris,trxs,xaij->aj", Dbar[s,s,o,s], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (18):", path_info[0])
        print("Path info for Eq. (18):", path_info[1])
    Cbar1[v,o] += np.einsum("tris,trxs,xaji->aj", Dbar[s,s,o,s], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:
        path_info = np.einsum_path("tris,trxs,xaji->aj", Dbar[s,s,o,s], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (18):", path_info[0])
        print("Path info for Eq. (18):", path_info[1])

    # Eq. (19)
    Cbar1[v,v] += 0.5 * np.einsum("klij,xaij,ybkl,xy->ab", Dbar[o,o,o,o], G[c,v,o,o], G[c,v,o,o], f[c,c], optimize="greedy")
    if debug:
        path_info = np.einsum_path("klij,xaij,ybkl,xy->ab", Dbar[o,o,o,o], G[c,v,o,o], G[c,v,o,o], f[c,c], optimize="greedy")
        print("Path info for Eq. (18):", path_info[0])
        print("Path info for Eq. (18):", path_info[1]) 
    Cbar1[v,v] += 0.5 * np.einsum("klij,xaij,xbkl->ab", Dbar[o,o,o,o], temp, G[c,v,o,o], optimize="greedy")
    if debug:
        path_info = np.einsum_path("klij,xaij,xbkl->ab", Dbar[o,o,o,o], temp, G[c,v,o,o], optimize="greedy")
        print("Path info for Eq. (18):", path_info[0])
        print("Path info for Eq. (18):", path_info[1])

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
    Cbar2 = np.zeros((nbf,nbf,nbf,nbf))
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
    Cbar2 = np.zeros((nbf,nbf,nbf,nbf))

    g_sscs=g.format_cbar1()

    # Eq. (27), without 0.5
    temp = np.einsum('xaij,aa->xaij', G[c,v,o,o], f[v,v])
    temp -= np.einsum('xaij,ii->xaij', G[c,v,o,o], f[o,o])
    temp -= np.einsum('xaij,jj->xaij', G[c,v,o,o], f[o,o])

    # Eq. (21)
    Cbar2[v,s,o,s] += 4 * np.einsum("ti,trxs,xaij->arjs", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:
        path_info = np.einsum_path("ti,trxs,xaij->arjs", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (21):", path_info[0])
        print("Path info for Eq. (21):", path_info[1])


    Cbar2[v,s,o,s] -= 2 * np.einsum("ti,trxs,xaji->arjs", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:  
        path_info = np.einsum_path("ti,trxs,xaji->arjs", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (21):", path_info[0])
        print("Path info for Eq. (21):", path_info[1])
    Cbar2[v,s,o,s] -= 2 * np.einsum("ti,rtxs,xaij->arjs", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:
        path_info = np.einsum_path("ti,rtxs,xaij->arjs", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (21):", path_info[0])
        print("Path info for Eq. (21):", path_info[1])
    


    Cbar2[s,v,o,o] += 4 * np.einsum("tu,ptxu,xaij->paij", D1[s,s], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:
        path_info = np.einsum_path("tu,ptxu,xaij->paij", D1[s,s], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (21):", path_info[0])
        print("Path info for Eq. (21):", path_info[1])
    Cbar2[s,v,o,o] -= 2 * np.einsum("tu,tpxu,xaij->paij", D1[s,s],  g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:
        path_info = np.einsum_path("tu,tpxu,xaij->paij", D1[s,s],  g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (21):", path_info[0])
        print("Path info for Eq. (21):", path_info[1])
    

    Cbar2[s,v,o,s] -= 2 * np.einsum("tj,ptxs,xaij->pais", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
    if debug:   
        path_info = np.einsum_path("tj,ptxs,xaij->pais", D1[s,o], g_sscs, G[c,v,o,o],optimize='greedy')
        print("Path info for Eq. (21):", path_info[0])
        print("Path info for Eq. (21):", path_info[1])
    # Eq. (22)
    Cbar2[o,v,v,o] += 2 * np.einsum("xaij,ybkl,xy,ki->labj", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o],optimize="greedy")
    if debug:   
        path_info = np.einsum_path("xaij,ybkl,xy,ki->labj", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o],optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])
    Cbar2[o,v,v,o] += 2 * np.einsum("xaij,xbkl,ki->labj", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
    if debug:
        path_info = np.einsum_path("xaij,xbkl,ki->labj", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])


    Cbar2[o,v,v,o] -= np.einsum("xaij,ybkl,xy,kj->labi", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o],optimize="greedy")
    if debug:
        path_info = np.einsum_path("xaij,ybkl,xy,kj->labi", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o],optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])
    Cbar2[o,v,v,o] -= np.einsum("xaij,xbkl,kj->labi", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
    if debug:
        path_info = np.einsum_path("xaij,xbkl,kj->labi", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])


    Cbar2[o,v,v,o] -= np.einsum("xaij,ybkl,xy,li->kabj", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o],optimize="greedy")
    if debug:
        path_info = np.einsum_path("xaij,ybkl,xy,li->kabj", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o],optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])
    Cbar2[o,v,v,o] -= np.einsum("xaij,xbkl,li->kabj", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
    if debug:
        path_info = np.einsum_path("xaij,xbkl,li->kabj", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])


    Cbar2[o,v,o,v] -= np.einsum("xaij,ybkl,xy,lj->kaib", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o], optimize="greedy")
    if debug:
        path_info = np.einsum_path("xaij,ybkl,xy,lj->kaib", G[c,v,o,o], G[c,v,o,o], f[c,c], D1[o,o], optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])
    Cbar2[o,v,o,v] -= np.einsum("xaij,xbkl,lj->kaib", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
    if debug:
        path_info = np.einsum_path("xaij,xbkl,lj->kaib", temp, G[c,v,o,o], D1[o,o],optimize="greedy")
        print("Path info for Eq. (22):", path_info[0])
        print("Path info for Eq. (22):", path_info[1])
    return Cbar2
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