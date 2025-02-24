"""
Run ct calculation then dump 
the integral in fci dump
"""
from ct.rhf_energy import rhf_energy
from ct.ct import canonical_transform
import psi4
import numpy as np
from psi4.core import MintsHelper
from write_dump import write_dump
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
bond_length = [2.0]

def do_ct(mol, r, b_basis, gamma):
    mol.r = r
    psi4.set_options({'basis': b_basis,
                      'df_basis_mp2': "cc-pVDZ-F12-Optri",
                      'scf_type': 'pk',
                      'maxiter': 40,
                      'screening': 'csam',
                      'e_convergence': 1e-10})
    e_hf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    mints=MintsHelper(wfn)
    ao_eri=mints.ao_eri().np
    mo_eri=mints.mo_eri(wfn.Ca(),wfn.Ca(),wfn.Ca(),wfn.Ca()).np
    mo_h_core=wfn.Ca().np.T@(mints.ao_potential().np+mints.ao_kinetic().np)@wfn.Ca().np
    V_nuc=psi_mol.nuclear_repulsion_energy()
    n_ele=wfn.nalpha()*2
    n_obs=wfn.nmo()
    write_dump("BARE.DUMP",n_ele=n_ele,n_obs=n_obs,V_nuc=V_nuc,mo_h_core=mo_h_core,mo_eri=mo_eri)
    basis = psi4.core.get_global_option('BASIS')
    df_basis = psi4.core.get_global_option('DF_BASIS_MP2')
    print("regular hf energy is ", e_hf)
    print("run ct scf")
    h_ct = canonical_transform(
        mol, wfn, basis, df_basis, gamma=gamma, frezee_core=True)
    rhf_ct = rhf_energy(psi_mol, wfn, h_ct)
    print("ct  hf energy is ", rhf_ct["escf"],
          " correlation energy is ", rhf_ct["escf"]-e_hf)
    h1e_ct, h2e_ct, cp_ct=convert_ct_to_quccsd(h_ct, rhf_ct)
    h1e_ct_mo=np.einsum("ij,iI,jJ->IJ",h1e_ct,cp_ct,cp_ct,optimize=True)
    h2e_ct_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",h2e_ct,cp_ct,cp_ct,cp_ct,cp_ct,optimize=True)
    h2_sym=(h2e_ct_mo+np.einsum("ijkl->jikl",h2e_ct_mo))/2
    write_dump("DRESSED.DUMP",n_ele=n_ele,n_obs=n_obs,V_nuc=V_nuc,mo_h_core=h1e_ct_mo,mo_eri=h2_sym)
    return e_hf,h_ct, rhf_ct


def convert_ct_to_quccsd(h_ct, hf_ct):
    h1e_ct = np.copy(h_ct['Hbar1'])
    h2e_ct = np.copy(h_ct['Hbar2'])
    h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)  # to ao basis
    cp_ct = hf_ct['C']
    return h1e_ct, h2e_ct, cp_ct



for r_h2 in bond_length:
    print("at bond length {}".format(r_h2))
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.clean_variables()
    e_hf,Hct, ct_rhf = do_ct(psi_mol, r_h2, B_BASIS, GAMMA)
    ct_escf = ct_rhf['escf']
    ## save integral
    h1e_ct, h2e_ct, cp_ct=convert_ct_to_quccsd(Hct,ct_rhf)

