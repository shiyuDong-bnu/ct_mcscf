""" This script is used to  calculate f12-hf
first do psi4 hf
then do ct
finally do ct-hf
"""

import sys

sys.path.append("/home/sydong/work/workspace/reproducing/final_ct/ct_mcscf/")
sys.path.append("/data/home/sydong/work/reproducing/forte")

from ct.hf_code.rhf_energy import rhf_energy
from ct.ct import canonical_transform
import fast_int
import psi4
#import forte
import numpy as np
from psi4.core import MintsHelper
from utils.write_dump import write_dump_np

# set geometry and basis
psi4.set_num_threads(56)
psi_mol = psi4.geometry(
    """
    Ne  0 0 0
    symmetry c1
    """
)
psi4.set_output_file("Ne_with_pt2.out", True)
GAMMA = 1.5
BASIS = ["aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz"]
#BASIS = ["aug-cc-pvdz-f12","aug-cc-pvtz-f12","aug-cc-pvqz-f12"]


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
    #        "df_basis_mp2": b_basis + "-Optri",
            "scf_type": "pk",
            "maxiter": 40,
            "freeze_core": True,
            "screening": "csam",
            "d_convergence": 1e-6,
            "e_convergence": 1e-10,
            "fast_int__gamma":1.5,
            "fast_int__n_active":0,
            "fast_int__n_restricted_docc":4,
            "fast_int__n_frozen_core":1,
            "fast_int__cabs_basis":b_basis+"-Optri",
            "df_basis_mp2":"aug-cc-pv6z-ri",
        }
    )

    e_hf, wfn = psi4.energy("scf", molecule=mol, return_wfn=True)
    #wfn=psi4.core.Wavefunction.from_file("f12mp2scf.npy")
    #e_hf= wfn.variable("CURRENT ENERGY")
    ## do fast int
    e,int_wfn=psi4.energy("fast_int",ref_wfn=wfn,return_wfn=True,)
    basis = psi4.core.get_global_option("BASIS")
    df_basis=psi4.core.get_local_option("FAST_INT","CABS_BASIS")
    print("regular hf energy is ", e_hf)
    print("run ct scf")
    print("frezze core", fc)
    h_ct = canonical_transform(mol, int_wfn,int_wfn, basis, df_basis, gamma=gamma, freeze_core=fc)
    rhf_ct = rhf_energy(psi_mol, int_wfn, h_ct)
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
