
import numpy as np
import psi4
import time
import ct.helper_HF as scf_helper
def rohf_energy(molecule, wfn, ct=None):
    maxiter = psi4.core.get_global_option('MAXITER')
    e_conv = psi4.core.get_global_option('E_CONVERGENCE')
    d_conv = psi4.core.get_global_option('D_CONVERGENCE')
    basis = psi4.core.get_global_option('BASIS')
    df_basis = psi4.core.get_global_option('DF_BASIS_MP2')

    ## Set occupations
    nocca = wfn.nalpha()
    noccb = wfn.nbeta()
    ndocc = min(nocca, noccb)
    nocc = max(nocca, noccb)
    nsocc = nocc - ndocc

    # assume close-shell without symmetry
    t = time.time()
    mints = wfn.mintshelper()
    S = wfn.S().to_array()
    nbf = S.shape[0]
    o = slice(0, ndocc)
    v = slice(ndocc, nbf)

    print('\nNumber of occupied orbitals: %d' % ndocc)
    print('Number of basis functions: %d' % nbf)
    print(' ms: ' ,nocca-noccb)

    if ct is None:
        H = wfn.H().to_array()
        I = mints.ao_eri().to_array().swapaxes(1,2)
        eri=mints.ao_eri()
        eshift = 0.0
    else:
        H = ct['Hbar1']
        I = ct['Hbar2']
        eri = I.swapaxes(1,2)
        # no2 = ndocc * ndocc
        # eshift = 2.0 * sum(np.diag(H[o,o]))
        # eshift += 2.0 * sum(np.diag(I[o,o,o,o].reshape(no2, no2)))
        # eshift -= sum(np.diag(I.transpose((0,1,3,2))[o,o,o,oHct].reshape(no2,no2)))

    print("\n  Total time taken for integrals %.3f seconds.\n" % (time.time()-t))
    t = time.time()

    # orthogonalize basis
    A = wfn.S().clone()
    A.power(-0.5, 1.e-16)
    A = A.to_array()

    # core guess
    guess="gwh"
    if guess == 'gwh':
        F = 0.875 * S * (np.diag(H)[:, None] + np.diag(H))
        F[np.diag_indices_from(F)] = np.diag(H)
    elif guess == 'core':
        F = H.copy()
    else:
        raise Exception("Unrecognized guess type %s. Please use 'core' or 'gwh'." % guess)
    Hp = A.dot(F).dot(A)
    eps, Ct = np.linalg.eigh(Hp)
    C = A.dot(Ct)
    Cnocc = C[:, :nocc]
    Docc = np.dot(Cnocc, Cnocc.T)
    Cndocc = C[:, :ndocc]
    Ddocc = np.dot(Cndocc, Cndocc.T)

    print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))
    t = time.time()
    E = 0.0
    Enuc = molecule.nuclear_repulsion_energy()
    Eold = 0.0
    
    diis = scf_helper.DIIS_helper()
    # scf iterations
    for SCF_ITER in range(1, maxiter + 1):
    

        # Build a and b fock matrices
        #J, K = scf_helper.compute_jk(jk, [C[:, :nocc], C[:, :ndocc]])
        ## build jk by hand
        D_o=np.einsum("iI,jI->ij",C[:,:nocc],C[:,:nocc])
        D_c=np.einsum("iI,jI->ij",C[:,:ndocc],C[:,:ndocc])
        J=[None,None]
        K=[None,None]
        J[0]=np.einsum("ijkl,kl->ij",eri,D_o)
        J[1]=np.einsum("ijkl,kl->ij",eri,D_c)
        K[0]=np.einsum("ijkl,jk->il",eri,D_o)
        K[1]=np.einsum("ijkl,jk->il",eri,D_c)
        J = J[0] + J[1]
        Fa = H + J - K[0]
        Fb = H + J - K[1]
    
        # Build MO Fock matrix
        moFa = (C.T).dot(Fa).dot(C)
        moFb = (C.T).dot(Fb).dot(C)
    
        # Special note on the ROHF Fock matrix (taken from Psi4)
        # Fo = open-shell fock matrix = 0.5 Fa
        # Fc = closed-shell fock matrix = 0.5 (Fa + Fb)
        #
        # The effective Fock matrix has the following structure
        #          |  closed     open    virtual
        #  ----------------------------------------
        #  closed  |    Fc     2(Fc-Fo)    Fc
        #  open    | 2(Fc-Fo)     Fc      2Fo
        #  virtual |    Fc       2Fo       Fc
    
        #print moFa[ndocc:nocc, ndocc:nocc] + moFb[ndocc:nocc, ndocc:nocc]
        moFeff = 0.5 * (moFa + moFb)
        moFeff[:ndocc, ndocc:nocc] = moFb[:ndocc, ndocc:nocc]
        moFeff[ndocc:nocc, :ndocc] = moFb[ndocc:nocc, :ndocc]
        moFeff[ndocc:nocc, nocc:] = moFa[ndocc:nocc, nocc:]
        moFeff[nocc:, ndocc:nocc] = moFa[nocc:, ndocc:nocc]
    
        # Back transform to AO Fock
        Feff = (Ct).dot(moFeff).dot(Ct.T)
    
        # Build gradient
        IFock = moFeff[:nocc, ndocc:].copy()
        IFock[:, :nsocc] /= 2
        IFock[ndocc:, :] /= 2
        IFock[ndocc:, :nsocc] = 0.0
    #    IFock[np.diag_indices_from(IFock)] = 0.0
        diis_e = (Ct[:, :nocc]).dot(IFock).dot(Ct[:, ndocc:].T)
        diis.add(Feff, diis_e)
    
        # SCF energy and update
        SCF_E  = np.einsum('pq,pq->', Docc + Ddocc, H)
        SCF_E += np.einsum('pq,pq->', Docc, Fa)
        SCF_E += np.einsum('pq,pq->', Ddocc, Fb)
        SCF_E *= 0.5
        SCF_E += Enuc
    
        dRMS = np.mean(diis_e**2)**0.5
        print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E'
              % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS))
        if (abs(SCF_E - Eold) < e_conv) and (dRMS < d_conv):
            break
    
        Eold = SCF_E
    
        # Build new orbitals
        Feff = diis.extrapolate()
        e, Ct = np.linalg.eigh(Feff)
        C = A.dot(Ct)
    
        Cnocc = C[:, :nocc]
        Docc = np.dot(Cnocc, Cnocc.T)
        Cndocc = C[:, :ndocc]
        Ddocc = np.dot(Cndocc, Cndocc.T)
    
        if SCF_ITER == maxiter:
            raise Exception("Maximum number of SCF cycles exceeded.")
    ## calculate eps and C using converged Fock matrix ,not extrapolate Fock.
    eps, C2 = np.linalg.eigh(Feff)
    C = A.dot(C2)
    print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))

    print('Final SCF energy: %.8f hartree' % SCF_E)

    print("Orbital Energy ",eps )
    out = {'wfn': wfn, 'C': C, 'eps': eps, 'H1': H, 'H2': I, 'nbf': nbf, 'ndocc': ndocc, 'escf': SCF_E}
    return out
