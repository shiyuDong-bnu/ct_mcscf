import sys
import numpy as np
import psi4


sys.path.append("/data/home/sydong/work/reproducing/forte")
import forte
from forte.proc.orbital_helpers import  ortho_orbs_forte
psi4.set_output_file("dressed1.out")
psi4.set_num_threads(64)
psi4.set_memory("16GB")
from point import gen_point

length=gen_point()

psi_mol = psi4.geometry(
    """
0 1
H 0 0 0
H 0 0 r_hh

symmetry c1
"""
)

GAMMA = 0.7
BASIS_SET = "cc-pvdz"
FREEZE_CORE = True
N_ACTIVE = 2
N_FROZEN = 0
N_RESTRICTED_DOCC = 0
N_OCC = N_FROZEN + N_RESTRICTED_DOCC

mos_spaces = {
    "active": [N_ACTIVE],  ## 2p
    "restricted_docc": [N_RESTRICTED_DOCC],
    "frozen_docc": [N_FROZEN],
}
psi_mol.r_nn = 1.5
data = forte.modules.ObjectsUtilPsi4(
    molecule=psi_mol, basis=BASIS_SET, mo_spaces=mos_spaces
).run()
mo_space_info = data.mo_space_info

psi4.set_options(
    {
        "basis": BASIS_SET,
        "reference": "rhf",
        # 'df_basis_mp2': BASIS_SET+"-mp2fit",
        "scf_type": "pk",
        "screening": "csam",
        "maxiter": 300,
        "e_convergence": 10,
        "d_convergence": 8,
    }
)

forte_options = {
    #"job_type":"new_driver",
    "active_space_solver": "fci",
    "active": [
        N_ACTIVE,
    ],  # 2p
    "restricted_docc": [N_RESTRICTED_DOCC],  # 2s
    "frozen_docc": [N_FROZEN],  # 2s
    ## this must include we do not opimize frozne orbital
    ## in after ct
    "MCSCF_IGNORE_FROZEN_ORBS": False,

    "DSRG_RDM_MS_AVG":True,
    "ci_spin_adapt": True,
    "DL_GUESS_PER_ROOT":4,
    "print": 3,

    "int_type": "fcidump",
    "fcidump_file":"BOND_1.5_MRDRESSED.DUMP",

    "correlation_solver": "sa-mrdsrg",
    "corr_level":"pt2",
    "dsrg_s": 0.5,

    "e_convergence"       :  1e-8 , # energy convergence of the FCI iterations
    "r_convergence":         1e-7 , # residual convergence of the FCI iterations
    "mcscf_e_convergence": 1e-8,  # energy convergence of the MCSCF iterations
    "mcscf_g_convergence": 1e-7,  # gradient convergence of the MCSCF iterations
    #"mcscf_micro_maxiter": 1,  # do at least 4 micro iterations per macro iteration
}
for i in length:
    psi_mol.r_nn = i
    forte_options["fcidump_file"] = f"BOND_{i:.2f}_MRDRESSED.DUMP"
    print(forte_options["fcidump_file"])
    e_ct_scf, ct_mcscf_wfn = psi4.energy(
        "forte", forte_options=forte_options, return_wfn=True
    )
