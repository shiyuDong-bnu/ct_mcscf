import numpy as np
import psi4
import time
from ct.diis import DIIS
def rhf_energy(molecule, wfn, ct=None):
    maxiter = psi4.core.get_global_option('MAXITER')
    e_conv = psi4.core.get_global_option('E_CONVERGENCE')
    d_conv = psi4.core.get_global_option('D_CONVERGENCE')
    basis = psi4.core.get_global_option('BASIS')
    df_basis = psi4.core.get_global_option('DF_BASIS_MP2')

    # assume close-shell without symmetry
    t = time.time()
    mints = wfn.mintshelper()
    S = wfn.S().to_array()
    nbf = S.shape[0]
    ndocc = wfn.nalpha()
    o = slice(0, ndocc)
    v = slice(ndocc, nbf)

    print('\nNumber of occupied orbitals: %d' % ndocc)
    print('Number of basis functions: %d' % nbf)

    if ct is None:
        H = wfn.H().to_array()
        I = mints.ao_eri().to_array().swapaxes(1,2)
        eshift = 0.0
    else:
        H = ct['Hbar1']
        I = ct['Hbar2']
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
    Hp = A.dot(H).dot(A)
    e, C2 = np.linalg.eigh(Hp)
    C = A.dot(C2)
    Cocc = C[:,o]
    D = np.einsum('pi,qi->pq', Cocc, Cocc)
    print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))
    t = time.time()

    # scf iterations
    enuc = molecule.nuclear_repulsion_energy()
    eold = 0.0
    Dold = np.zeros_like(D)
    ## set up diis
    my_diis=DIIS()
    for scf_iter in range(1, maxiter + 1):
        J = np.einsum('pqrs,qs->pr', I, D)
        K = np.einsum('pqrs,qr->ps', I, D)
        F = H + 2 * J - K
    
        
        scf_e = np.einsum('pq,pq->', F + H, D) + enuc
        drms = np.sum(np.power(D - Dold, 2)) ** 0.5

        diis_error=my_diis.calc_error(F,D,S,A)
        my_diis.update(F,diis_error)

        print('SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E' % (scf_iter, scf_e, (scf_e - eold), drms))
        if (abs(scf_e - eold) < e_conv) and (drms < d_conv):
            break
    
        eold = scf_e
        Dold = D

        if scf_iter >1:
            F=my_diis.extropolate()
        Fp = A.dot(F).dot(A)
        e, C2 = np.linalg.eigh(Fp)
        C = A.dot(C2)
        Cocc = C[:,o]
        D = np.einsum('pi,qi->pq', Cocc, Cocc)
    
        if scf_iter == maxiter:
            psi4.core.clean()
            raise Exception("Maximum number of SCF cycles exceeded.")
    
    print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))
    
    print('Final SCF energy: %.8f hartree' % scf_e)
    
    # build fock and get orbital energies
    Fmo = C.T.dot(F).dot(C)
    eps, C2 = np.linalg.eigh(Fp)
    C = A.dot(C2)
    print(f'Orbital energies: {eps}')

    out = {'wfn': wfn, 'C': C, 'eps': eps, 'H1': H, 'H2': I, 'nbf': nbf, 'ndocc': ndocc, 'escf': scf_e}
    return out
