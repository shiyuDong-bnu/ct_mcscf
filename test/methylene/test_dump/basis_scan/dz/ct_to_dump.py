import sys
import psi4
import numpy as np
sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/software/forte")
from ct.mrct import canonical_transform
from ct.rhf_energy import rhf_energy
from utils.write_dump import write_dump_np

psi4.set_output_file("gen_dump.out")

psi_mol=psi4.geometry(
    """
0 1
C
H 1 r_ch
H 1 r_ch 2 ahch
unit bohr
symmetry c1"""
)
psi_mol.r_ch=2.1023
psi_mol.ahch=101.71

GAMMA = 1.0 
BASIS_SET="cc-pVDZ-F12"
FREEZE_CORE=True

psi4.set_options(
    {
        "basis":BASIS_SET,
        "reference":"rhf",
        'df_basis_mp2': BASIS_SET+"-Optri",
        "scf_type":"pk",
        'screening': 'csam',
        "maxiter":300,
        "e_convergence":10,
        "d_convergence":8,
        "freeze_core":True,
        # # "RESTRICTED_DOCC":[1,0,0,0],
        # "ACTIVE":[3,0,1,2],
    }
)

forte_mcscf_options={
    "job_type":"mcscf_two_step",
    "active_space_solver":"fci",
    "active":[6],
    "frozen_docc":[1],
    "MCSCF_IGNORE_FROZEN_ORBS":False,

}

## do mcscf calculation to generate 1RDM and 2RDM (density matrix)
e_hf,hf_wfn=psi4.energy("scf",return_wfn=True)
e_mcscf,mcscf_wfn=psi4.energy("forte",forte_options=forte_mcscf_options,
        ref_wfn=hf_wfn,return_wfn=True)
print("hf energy",e_hf)
print("mcscf energy",e_mcscf)
basis = psi4.core.get_global_option('BASIS')
df_basis = psi4.core.get_global_option('DF_BASIS_MP2')
mr_info={}
mr_info['occupied_index']=slice(0,1)
mr_info['active_index']=slice(1,7)
mr_info['RDM1']=psi4.core.variable("CURRENT RDM1").np
mr_info['RDM2']=psi4.core.variable("CURRENT RDM2").np.reshape((6,)*4)
## do ct to get integrals
h_ct = canonical_transform(
    psi_mol, mcscf_wfn, basis, df_basis, gamma=GAMMA, freeze_core=FREEZE_CORE,mr_info=mr_info)
## do ct hf to get MO coeff
rhf_ct = rhf_energy(psi_mol, mcscf_wfn, h_ct)
print("ct  hf energy is ", rhf_ct["escf"],
        " correlation energy is ", rhf_ct["escf"]-e_hf)
## dump ct hamiltonian
def convert_ct_to_quccsd(hamiltonian, hf_ct):
    """ this function just uppack integral
    and change the eri from physical to chemical order"""

    h1 = np.copy(hamiltonian['Hbar1'])
    h2 = np.copy(hamiltonian['Hbar2'])
    h2 = np.einsum("ijkl->ikjl", h2)  
    cp = hf_ct['C']
    return h1, h2 ,cp

n_ele=mcscf_wfn.nalpha()*2
n_obs=mcscf_wfn.nmo()
ms2=mcscf_wfn.nalpha()-mcscf_wfn.nbeta()
V_nuc=psi_mol.nuclear_repulsion_energy()
h1e_ct, h2e_ct, cp_ct=convert_ct_to_quccsd(h_ct, rhf_ct)
h1e_ct_mo=np.einsum("ij,iI,jJ->IJ",h1e_ct,cp_ct,cp_ct,optimize=True)
h2e_ct_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",h2e_ct,cp_ct,cp_ct,cp_ct,cp_ct,optimize=True)
write_dump_np("MRDRESSED.DUMP",n_ele=n_ele,n_obs=n_obs,V_nuc=V_nuc,ms2=ms2,
        mo_h_core=h1e_ct_mo,mo_eri=h2e_ct_mo)

