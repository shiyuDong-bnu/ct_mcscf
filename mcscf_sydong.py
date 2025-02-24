import veloxchem as vlx
import multipsi as mtp
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

mol_template = """
Li 0.0000 0.0000  0.0000
Li 0.0000 0.0000  bond_length
"""
scf_drv = vlx.ScfRestrictedDriver()

mol_str = mol_template.replace("bond_length", str(2.0))
molecule = vlx.Molecule.read_molecule_string(mol_str, units='angstrom')
basis = vlx.MolecularBasis.read(molecule, "cc-pvdz")
## this is ao integrals
nbas=vlx.MolecularBasis.get_dimension_of_basis(basis,molecule)
V_nuc=molecule.nuclear_repulsion_energy()
kinetic_drv=vlx.KineticEnergyIntegralsDriver()
T=kinetic_drv.compute(molecule,basis).to_numpy()
nucpot_drv=vlx.NuclearPotentialIntegralsDriver()
V=-nucpot_drv.compute(molecule,basis).to_numpy()
h_ao=T+V
eri_drv=vlx.ElectronRepulsionIntegralsDriver()
g_ao=np.zeros((nbas,)*4)
eri_drv.compute_in_memory(molecule,basis,g_ao)

## do an scf
scf_results = scf_drv.compute(molecule, basis)

## calculate CI ,and get RDMS
space=mtp.OrbSpace(molecule, scf_drv.mol_orbs) # MolSpace will store the orbitals for us
space.cas(2,4)
nIn = space.n_inactive # Number of inactive orbitals
nAct = space.n_active # Number of active orbitals
active_index=slice(nIn,nIn+nAct)
silent = vlx.OutputStream(None) # Deactivate printing
CIdrv = mtp.CIDriver(ostream=silent)
## from here macro iteration
CONVERGE=False
GNORM=1e-5
COUNT=0
while not CONVERGE:
    print("At Iteration {}".format(COUNT+1))
    ## perpare for macro iteration
    mo_coeff = space.molecular_orbitals.alpha_to_numpy()

    C_inact=mo_coeff[:,:nIn]
    C_act=mo_coeff[:,active_index]

    Density_inac=np.einsum("iI,jI->ij",C_inact,C_inact)
    J_in_ao=np.einsum("ijkl,kl->ij",g_ao,Density_inac)
    K_in_ao=np.einsum("ijkl,jk->il",g_ao,Density_inac)
    ## equ(156) F_{pi}^A
    F_inactive_ao=h_ao+2*J_in_ao-K_in_ao
    E_inactive = np.einsum('ij,ij->', h_ao + F_inactive_ao, Density_inac) + V_nuc
    # Transform to MO basis
    h_active_mo = np.einsum("pq, qu, pt->tu", F_inactive_ao, C_act, C_act)
    ## generate active integral and inactive energy and fock
    g_active_mo_int=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",g_ao,C_act,C_act,
                            C_act,C_act)
    # Obtain the integrals needed for CI
    CIdrv._update_integrals(E_inactive, h_active_mo, g_active_mo_int)
    ci_results = CIdrv.compute(molecule, basis, space) #Compute only the ground state
    E_ci = CIdrv.get_energy()
    D_act=CIdrv.get_active_density()
    Gamma_act=CIdrv.get_active_2body_density(0)
    np.save("gamma.npy",Gamma_act)
    ##  prepare for micro iteration
    Density_active_ao=np.einsum("iI,jJ,IJ->ij",C_act,C_act,
                                D_act,optimize=True)
    J_act_ao=np.einsum("ijkl,kl->ij",g_ao,Density_active_ao,optimize=True)
    K_act_ao=np.einsum("ijkl,kj->il",g_ao,Density_active_ao,optimize=True)
    F_active_ao=J_act_ao-0.5*K_act_ao
    ## transform to mo
    ## ## equation 156
    F_pi_I=np.einsum("ij,iI,jJ->IJ",F_inactive_ao,mo_coeff,C_inact) 
    ## ## equation 157
    F_pi_A=np.einsum("ij,iI,jJ->IJ",F_active_ao,mo_coeff,C_inact)
    ## equation 155
    F_ip=2*(F_pi_I+F_pi_A).T
    ## then active genearl part
    F_pv_I=np.einsum("ij,iI,jJ->IJ",F_inactive_ao,mo_coeff,C_act,
            optimize=True)
    F_up=np.einsum("pv,uv->pu",F_pv_I,D_act).T
    g_pwvx=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",g_ao,mo_coeff,
                     C_act,C_act,C_act,optimize=True)
    ## equation 167
    Q_pu=np.einsum("pwvx,uwvx->pu",g_pwvx,Gamma_act)
    ## equation 166
    F_up+=Q_pu.T
    F_total=np.zeros((nbas,nbas))
    F_total[:nIn,:]=F_ip
    F_total[active_index,:]=F_up
    ## equation 147
    kappa=2*(F_total-F_total.T)
    print("Energy is {:.10f},grad norm {:.6f}".format(E_ci,np.linalg.norm(kappa.flatten())))
    diagnal1 = np.einsum("pq,pm,qm->m", 2*(F_inactive_ao+F_active_ao), mo_coeff, mo_coeff
                         ,optimize=True) # Diagonal of 2*Fin+Fact in MO basis
    diagnal2 = np.diagonal(F_total) # Diagonal of the effective Fock matrix

    # Form Hessian diagonal
    diga_hess = np.zeros((nbas,nbas))
    diga_hess[:nIn,nIn:] = 2* diagnal1[nIn:] - 2* diagnal1[:nIn].reshape(-1,1) #Sum of a line and column vectors
    diga_hess[nIn:nIn+nAct,:] = - 2 * diagnal2[nIn:nIn+nAct].reshape(-1, 1)
    diga_hess[nIn:nIn+nAct,nIn+nAct:] += np.einsum('tt,a->ta',D_act,diagnal1[nIn+nAct:])
    diga_hess[nIn:nIn+nAct,:nIn] += np.einsum('tt,a->ta',D_act,diagnal1[:nIn])
    diga_hess += np.transpose(diga_hess)
    diga_hess[:nIn,:nIn] = 1 # To avoid division by 0
    diga_hess[nIn + nAct:, nIn + nAct:] = 1 # To avoid division by 0
    step=kappa/diga_hess
    ## equation 126
    U_tran=scipy.linalg.expm(step)
    new_mo_coeff=np.einsum("iI,IJ->iJ",mo_coeff,U_tran)
    ## reset orbitals 
    ene = np.zeros(nbas)
    occ = np.zeros(nbas)
    newmolorb = vlx.MolecularOrbitals([new_mo_coeff], [ene], [occ], vlx.molorb.rest)
    space.molecular_orbitals = newmolorb
    CONVERGE=np.linalg.norm(kappa.flatten())<GNORM
    COUNT+=1
