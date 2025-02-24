"""
run potential surface scan using both ct and nonct
and save the results
"""
from ct.rhf_energy import rhf_energy
from ct.ct import canonical_transform
import scipy.linalg
import psi4
import numpy as np
import multipsi as mtp
import veloxchem as vlx
import matplotlib.pyplot as plt
import scipy.linalg
import veloxchem as vlx
import multipsi as mtp
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
# set geometry and basis
psi_mol = psi4.geometry(
    """
    Li  0 0 0
    Li 0 0 r
    symmetry c1
    """
)
psi4.core.clean_options()
psi4.core.clean()
psi4.core.clean_variables()
psi4.set_output_file("ct_ucc_test.out", True)
B_BASIS = "cc-pvdz"
GAMMA = 0.6
#bond_length = np.concatenate([np.linspace(1.5, 2.5, 7),
#                             np.linspace(2.5, 3.0, 5),
#                             np.linspace(3.0, 5.0, 5)])
bond_length=[2.0]
E_HF=[]
E_MCSCF=[]
E_CT=[]
E_CT_MCSCF=[]


def do_ct(mol, r, b_basis, gamma):
    mol.r = r
    psi4.set_options({'basis': b_basis,
                      'df_basis_mp2': "cc-pVDZ-F12-Optri",
                      'scf_type': 'pk',
                      'maxiter': 40,
                      'screening': 'csam',
                      'e_convergence': 1e-10})
    e_hf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    basis = psi4.core.get_global_option('BASIS')
    df_basis = psi4.core.get_global_option('DF_BASIS_MP2')
    print("regular hf energy is ", e_hf)
    print("run ct scf")
    h_ct = canonical_transform(
        mol, wfn, basis, df_basis, gamma=gamma, frezee_core=True)
    rhf_ct = rhf_energy(psi_mol, wfn, h_ct)
    print("ct  hf energy is ", rhf_ct["escf"],
          " correlation energy is ", rhf_ct["escf"]-e_hf)
    return e_hf,h_ct, rhf_ct


def convert_ct_to_quccsd(h_ct, hf_ct):
    h1e_ct = np.copy(h_ct['Hbar1'])
    h2e_ct = np.copy(h_ct['Hbar2'])
    h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)  # to ao basis
    cp_ct = hf_ct['C']
    return h1e_ct, h2e_ct, cp_ct

def run_mcscf(r_h2,h1e_ct, h2e_ct, cp_ct,DO_CT=True):
    mol_template = """
    Li 0.0000 0.0000  0.0000
    Li 0.0000 0.0000  bond_length
    """
    scf_drv = vlx.ScfRestrictedDriver()

    mol_str = mol_template.replace("bond_length", str(r_h2))
    molecule = vlx.Molecule.read_molecule_string(mol_str, units='angstrom')
    basis = vlx.MolecularBasis.read(molecule, B_BASIS)
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

    ## load ct integral
    g_ao_ct=h2e_ct
    h_ao_ct=h1e_ct
    C_ao_ct=cp_ct
    if DO_CT:
        #g_ao=g_ao_ct
        g_ao=(g_ao_ct+np.einsum("ijkl->jikl",g_ao_ct))/2
        h_ao=h_ao_ct
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
        if COUNT==0 and DO_CT:
            mo_coeff=cp_ct
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
    return E_ci



for r_h2 in bond_length:
    print("at bond length {}".format(r_h2))
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.clean_variables()
    e_hf,Hct, ct_rhf = do_ct(psi_mol, r_h2, B_BASIS, GAMMA)
    ct_escf = ct_rhf['escf']

    
    ## save integral
    h1e_ct, h2e_ct, cp_ct=convert_ct_to_quccsd(Hct,ct_rhf)
    e_mcscf=run_mcscf(r_h2,h1e_ct, h2e_ct, cp_ct,False)
    e_ct_mcscf=run_mcscf(r_h2,h1e_ct, h2e_ct, cp_ct,True)
    E_HF.append(e_hf)
    E_MCSCF.append(e_mcscf)
    E_CT.append(ct_escf)
    E_CT_MCSCF.append(e_ct_mcscf)
    print(e_hf)
    print(e_mcscf)
    print(ct_escf)
    print(e_ct_mcscf)
#np.save("e_hf.npy",E_HF)
#np.save("e_mcscf.npy",E_MCSCF)
#np.save("e_ct.npy",E_CT)
#np.save("e_ct_mcscf.npy",E_CT_MCSCF)
