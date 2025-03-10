"""
Run ct calculation then dump 
the integral in fci dump
calculate using forte
"""
from ct.rhf_energy import rhf_energy
from ct.ct import canonical_transform
import psi4
import numpy as np
from psi4.core import MintsHelper
from write_dump import write_dump_np
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
bond1=np.linspace(3.0,4.0,5)
bond2=np.linsapce
bond_length = [2.0]

def do_ct(mol, r, b_basis, gamma,fc=True):
    mol.r = r
    psi4.set_options({'basis': b_basis,
                      'df_basis_mp2': "cc-pVDZ-F12-Optri",
                      'scf_type': 'pk',
                      'maxiter': 40,
                      'screening': 'csam',
                      'e_convergence': 1e-10})
    e_hf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    mints=MintsHelper(wfn)
    mo_eri=mints.mo_eri(wfn.Ca(),wfn.Ca(),wfn.Ca(),wfn.Ca()).np
    mo_h_core=wfn.Ca().np.T@(mints.ao_potential().np+mints.ao_kinetic().np)@wfn.Ca().np
    V_nuc=psi_mol.nuclear_repulsion_energy()
    n_ele=wfn.nalpha()*2
    n_obs=wfn.nmo()
    write_dump_np("BARE.DUMP",n_ele=n_ele,n_obs=n_obs,V_nuc=V_nuc,mo_h_core=mo_h_core,mo_eri=mo_eri)
    basis = psi4.core.get_global_option('BASIS')
    df_basis = psi4.core.get_global_option('DF_BASIS_MP2')
    print("regular hf energy is ", e_hf)
    print("run ct scf")
    print("frezze core",fc)
    h_ct = canonical_transform(
        mol, wfn, basis, df_basis, gamma=gamma, frezee_core=fc)
    rhf_ct = rhf_energy(psi_mol, wfn, h_ct)
    print("ct  hf energy is ", rhf_ct["escf"],
          " correlation energy is ", rhf_ct["escf"]-e_hf)
    h1e_ct, h2e_ct, cp_ct=convert_ct_to_quccsd(h_ct, rhf_ct)
    h1e_ct_mo=np.einsum("ij,iI,jJ->IJ",h1e_ct,cp_ct,cp_ct,optimize=True)
    h2e_ct_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",h2e_ct,cp_ct,cp_ct,cp_ct,cp_ct,optimize=True)
    write_dump_np("DRESSED.DUMP",n_ele=n_ele,n_obs=n_obs,V_nuc=V_nuc,mo_h_core=h1e_ct_mo,mo_eri=h2e_ct_mo)
    return e_hf,h_ct, rhf_ct


def convert_ct_to_quccsd(h_ct, hf_ct):
    h1e_ct = np.copy(h_ct['Hbar1'])
    h2e_ct = np.copy(h_ct['Hbar2'])
    h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)  # to ao basis
    cp_ct = hf_ct['C']
    return h1e_ct, h2e_ct, cp_ct

def forte_calc(mol,r,dump_fname):
    mol.r=r
   # psi4.core.clean_options()
   # psi4.core.clean()
   # psi4.core.clean_variables()
    import forte
    psi4.set_options(
        {
     #       "basis":"cc-pvdz",
     #       "reference":"rhf",
     #       "scf_type":"pk",
            ## set forte
            #"forte__job_type":"mcscf_two_step",
            "forte__active_space_solver":"fci",
            "forte__restricted_docc":[2],
            "forte__active":[4],
            "forte__int_type" :"fcidump",
            "forte__fcidump_file":dump_fname,
            "forte__mcscf_g_convergence":6,
            "forte__mcscf_micro_maxiter":4,
            "forte__correlation_solver":"mrdsrg",
            "forte__corr_level":"ldsrg2",
            "forte__dsrg_s":0.5,
        }
    )
    psi4.energy("forte")
    variables=psi4.core.variables()
    e_ref=variables['DSRG REFERENCE ENERGY']
    e_final=variables['CURRENT ENERGY']
    return  e_ref,e_final    

for r_h2 in bond_length:
    print("at bond length {}".format(r_h2))
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.clean_variables()
    e_hf,Hct, ct_rhf = do_ct(psi_mol, r_h2, B_BASIS, GAMMA,fc=True)
    ct_escf = ct_rhf['escf']
    e_mcscf_bare,e_corre_bare=forte_calc(psi_mol,r_h2,dump_fname="BARE.DUMP")
    e_mcscf_dressed,e_corre_dressed=forte_calc(psi_mol,r_h2,dump_fname="DRESSED.DUMP")
    print("hatree fock",e_hf)
    print("ct hf",ct_escf)
    print("bare mcscf",e_mcscf_bare)
    print("bare corr",e_corre_bare)
    print("dressed mcscf",e_mcscf_dressed)
    print("dressed corr",e_corre_dressed)


