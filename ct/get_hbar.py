import numpy as np
def get_hbar(my_orbital_space,V_rational,X_rational,B_rational,D1,D2,g,G,f,h):
    s=my_orbital_space.s
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v

    nbf=my_orbital_space.nbf
    V = np.zeros((nbf, nbf, nbf, nbf))
    V[s,s,o,o] = V_rational
    del V_rational
    # Eq. (24)
    X = np.zeros((nbf, nbf, nbf, nbf))
    X[o,o,o,o] = X_rational
    del X_rational
    # Eq. (25)
    B = np.zeros((nbf, nbf, nbf, nbf))
    B[o,o,o,o] =B_rational
    del B_rational
    # Eq. (26)
    U = np.einsum("prxs,xbij->prsijb", g[s,s,c,s], G[c,v,o,o])

    # Eq. (27), without 0.5
    S = np.einsum("xaij,ybkl,xy->klbija", G[c,v,o,o], G[c,v,o,o], f[c,c], optimize=True)
    temp = np.einsum('xaij,aa->xaij', G[c,v,o,o], f[v,v])
    temp -= np.einsum('xaij,ii->xaij', G[c,v,o,o], f[o,o])
    temp -= np.einsum('xaij,jj->xaij', G[c,v,o,o], f[o,o])
    S += np.einsum("xaij,xbkl->klbija", temp, G[c,v,o,o])

    # Eq. (28)
    Dbar = 2 * np.einsum("pq,rs->prqs", D1, D1) - np.einsum("ps,rq->prqs", D1, D1) - D2
    del D2,temp

    # 1-body
    Cbar1 = np.zeros((nbf,nbf))

    # Eq. (18)
    Cbar1[v,s] += np.einsum("trij,trqija->aq", Dbar[s,s,o,o], U)

    Ut = 2 * np.copy(U) - U.transpose((0,1,2,4,3,5))
    Cbar1[v,o] -= np.einsum("tris,trsija->aj", Dbar[s,s,o,s], Ut)

    # Eq. (19)
    Cbar1[v,v] += 0.5 * np.einsum("klij,klbija->ab", Dbar[o,o,o,o], S)

    # Eq. (15)
    hbar = h[s,s] + 0.5 * Cbar1[s,s] + 0.5 * Cbar1.T

    # 2-body
    Cbar2 = np.zeros((nbf,nbf,nbf,nbf))

    # Eq. (21)
    Ut -= U.transpose((1,0,2,3,4,5))
    Cbar2[v,s,o,s] += 2 * np.einsum("ti,trsija->arjs", D1[s,o], Ut)

    Ut = 2 * U - U.transpose((1,0,2,3,4,5))
    Cbar2[s,v,o,o] += 2 * np.einsum("tu,ptuija->paij", D1[s,s], Ut)

    Cbar2[s,v,o,s] -= 2 * np.einsum("tj,ptsija->pais", D1[s,o], U)
    del Ut,U
    # Eq. (22)
    Cbar2[o,v,v,o] += 2 * np.einsum("klbija,ki->labj", S, D1[o,o])
    Cbar2[o,v,v,o] -= np.einsum("klbija,kj->labi", S, D1[o,o])
    Cbar2[o,v,v,o] -= np.einsum("klbija,li->kabj", S, D1[o,o])
    Cbar2[o,v,o,v] -= np.einsum("klbija,lj->kaib", S, D1[o,o])

    # Eq. (20)
    Cbar2[s,v,o,o] += 4 * np.einsum("px,xbij->pbij", h[s,c], G[c,v,o,o])
    Cbar2[s,s,o,o] += 2 * V[s,s,o,o]
    Cbar2[s,o,o,o] -= 2 * np.einsum("klij,pk->plij", X[o,o,o,o], f[s,o])
    Cbar2[o,o,o,o] += B[o,o,o,o]

    # Eq. (16)
    gbar = 0.25 * (np.copy(Cbar2) + Cbar2.transpose((1,0,3,2)))
    gbar += 0.25 * (Cbar2.transpose((2,3,0,1)) + Cbar2.transpose((3,2,1,0)))
    gbar += g[s,s,s,s]
    return hbar ,gbar
