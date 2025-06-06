"""
This script is used to  calculate f12-hf
first do psi4 hf 
then do ct
finally do ct-hf
"""

import sys

sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/work/reproducing/forte")

from ct.rhf_energy import rhf_energy
from ct.ct import canonical_transform
import psi4
import forte
import numpy as np
from psi4.core import MintsHelper
from utils.write_dump import write_dump_np

# set geometry and basis
psi_mol = psi4.geometry(
    """
    Ne  0 0 0
    symmetry c1
    """
)
# psi4.core.clean_options()
# psi4.core.clean()
# psi4.core.clean_variables()
psi4.set_output_file("Ne_with_pt2.out", True)
GAMMA = 1.5
BASIS = ["aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz"]


def convert_ct_to_quccsd(h_ct, hf_ct):
    """
    This function just change the order of integral
    the function name is a legacy.
    """
    h1e_ct = np.copy(h_ct["Hbar1"])
    h2e_ct = np.copy(h_ct["Hbar2"])
    h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)  # to ao basis
    cp_ct = hf_ct["C"]
    return h1e_ct, h2e_ct, cp_ct


def do_ct(mol, b_basis, gamma, fc=True):
    psi4.set_options(
        {
            "basis": b_basis,
            "df_basis_mp2": b_basis + "-Optri",
            "scf_type": "pk",
            "maxiter": 40,
            "freeze_core": True,
            "screening": "csam",
            "d_convergence": 1e-6,
            "e_convergence": 1e-10,
        }
    )

    e_hf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    basis = psi4.core.get_global_option("BASIS")
    df_basis = psi4.core.get_global_option("DF_BASIS_MP2")
    print("regular hf energy is ", e_hf)
    print("run ct scf")
    print("frezze core", fc)
    h_ct = canonical_transform(mol, wfn, basis, df_basis, gamma=gamma, freeze_core=fc)
    rhf_ct = rhf_energy(psi_mol, wfn, h_ct)
    print(
        "ct  hf energy is ",
        rhf_ct["escf"],
        " correlation energy is ",
        rhf_ct["escf"] - e_hf,
    )
    return e_hf, h_ct, rhf_ct


for bs in BASIS:
    psi4.core.clean_options()
    psi4.core.clean()
    psi4.core.clean_variables()
    e_hf, Hct, ct_rhf = do_ct(psi_mol, bs, GAMMA, fc=True)
    ct_escf = ct_rhf["escf"]
    print(bs, " ", ct_escf)
