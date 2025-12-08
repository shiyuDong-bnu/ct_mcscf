"""
ct to dump has three parts
first parts is setting part which we set geometry , active space , options ets
second parts is calculation parts
  a. scf calculation
  b. casscf calculation
  c. canonical transformation parts
  d. save hamiltonian parts
"""
import sys

sys.path.append("/home/sydong/work/workspace/reproducing/final_ct/ct_mcscf/")
sys.path.append("/data/home/sydong/work/reproducing/forte")

from ct.mrct import canonical_transform
from ct.hf_code.rhf_energy import rhf_energy
from ct.utils.mr_info import MRInfo
from utils.write_dump import write_dump_np

import fast_int
import psi4
import numpy as np

psi4.set_output_file("dz_dump.out")
psi4.set_num_threads(64)
psi_mol = psi4.geometry(
    """
0 1
C
H 1 r_ch
H 1 r_ch 2 ahch
unit bohr
symmetry c1"""
)
psi_mol.r_ch = 2.1023
psi_mol.ahch = 101.71

GAMMA = 1.0
BASIS_SET = "cc-pVDZ-F12"
FREEZE_CORE = True
N_FROZEN = 1
N_RESTRICTED_DOCC = 0
N_ACTIVE = 6
N_OCC = N_FROZEN + N_RESTRICTED_DOCC

psi4.set_options(
    {
        "basis": BASIS_SET,
#        "reference": "hf",
        "scf_type": "pk",
        "freeze_core": True,
        "screening": "csam",
        "maxiter": 300,
        "e_convergence": 10,
        "d_convergence": 8,
        "fast_int__gamma": GAMMA,
        "fast_int__cabs_basis": BASIS_SET + "-Optri",
        "df_basis_mp2": BASIS_SET + "-mp2fit",
        "fast_int__n_active": N_ACTIVE,
        "fast_int__n_restricted_docc": N_RESTRICTED_DOCC,
        "fast_int__n_frozen_core": N_FROZEN,
    }
)
forte_mcscf_options = {
    "job_type": "mcscf_two_step",
    "active_space_solver": "fci",
    "restricted_docc": [N_RESTRICTED_DOCC],
    "active": [N_ACTIVE],
    "frozen_docc": [N_FROZEN],
}
e_hf, hf_wfn = psi4.energy("scf", return_wfn=True)
hf_wfn.write_molden("hf.molden")
e_mcscf, mcscf_wfn = psi4.energy(
    "forte", forte_options=forte_mcscf_options, ref_wfn=hf_wfn, return_wfn=True
)
mcscf_wfn.write_molden("cas.molden")
orbinf = {
    "n_frozen": N_FROZEN,
    "n_restricted_docc": N_RESTRICTED_DOCC,
    "n_active": N_ACTIVE,
    "n_virtual": mcscf_wfn.nmo() - (N_FROZEN + N_RESTRICTED_DOCC + N_ACTIVE),
}
mr_info = MRInfo(
    orbinf=orbinf,
    rdm1=psi4.core.variable("CURRENT RDM1").np,
    rdm2=psi4.core.variable("CURRENT RDM2").np.reshape((N_ACTIVE,) * 4),
)
## calculate int begin
e, int_wfn = psi4.energy(
    "fast_int",
    ref_wfn=mcscf_wfn,
    return_wfn=True,
)
basis = psi4.core.get_global_option("BASIS")
df_basis = psi4.core.get_local_option("FAST_INT", "CABS_BASIS")
## ct begin
h_ct = canonical_transform(
    psi_mol,
    mcscf_wfn,
    int_wfn,
    basis,
    df_basis,
    gamma=GAMMA,
    freeze_core=FREEZE_CORE,
    mr_info=mr_info,
)
## ct-hf begin
rhf_ct = rhf_energy(psi_mol, mcscf_wfn, h_ct)
print(
    "ct  hf energy is ",
    rhf_ct["escf"],
    " correlation energy is ",
    rhf_ct["escf"] - e_hf,
)
## save hamiltonina begin
n_ele = mcscf_wfn.nalpha() + mcscf_wfn.nbeta()
n_obs = mcscf_wfn.nmo()
ms2 = mcscf_wfn.nalpha() - mcscf_wfn.nbeta()
V_nuc = psi_mol.nuclear_repulsion_energy()
print("Total electron", n_ele)
print("Total orbital", n_obs)
print("MS2 ", ms2)
h1e_ct = h_ct["Hbar1"]
h2e_ct = h_ct["Hbar2"]  ## the returned ct 2eri is physical
h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)
cp_ct = mcscf_wfn.Ca()
h1e_ct_mo = np.einsum("ij,iI,jJ->IJ", h1e_ct, cp_ct, cp_ct, optimize=True)
h2e_ct_mo = np.einsum(
    "ijkl,iI,jJ,kK,lL->IJKL", h2e_ct, cp_ct, cp_ct, cp_ct, cp_ct, optimize=True
)
write_dump_np(
    "MRDRESSED.DUMP",
    n_ele=n_ele,
    n_obs=n_obs,
    V_nuc=V_nuc,
    ms2=ms2,
    mo_h_core=h1e_ct_mo,
    mo_eri=h2e_ct_mo,
)
