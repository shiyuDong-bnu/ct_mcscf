import sys
import numpy as np

sys.path.append("/home/sydong/work/workspace/reproducing/final_ct/ct_mcscf/")
sys.path.append("/data/home/sydong/work/reproducing/forte")

from utils.write_dump import write_dump_np

import psi4
psi4.set_output_file("dressed.out")

psi_mol = psi4.geometry(
    """
0 1
C
H 1 r_ch
H 1 r_ch 2 ahch
unit bohr
symmetry c1
"""
)

psi_mol.r_ch = 2.1023
psi_mol.ahch = 101.71

GAMMA = 1.0
BASIS_SET="cc-pVDZ-F12"

N_FROZEN=1
N_RESTRICTED_DOCC=0
N_ACTIVE=6
N_OCC=N_FROZEN+N_RESTRICTED_DOCC
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
        }
    )

forte_options={
  "active_space_solver":                "fci",
  "active"         :       [N_ACTIVE,],  # 2p
  "restricted_docc":[N_RESTRICTED_DOCC], # 2s
  "frozen_docc":[N_FROZEN],   #2s
  "MCSCF_IGNORE_FROZEN_ORBS":False,

  "int_type"         :     "fcidump",
  "fcidump_file"      :   "MRDRESSED.DUMP",
  "correlation_solver":"dsrg-mrpt2",


  "ci_spin_adapt": True,
  "DL_MAXITER":200,
   "MCSCF_MCI_MAXITER":100,
   "MCSCF_MICRO_MAXITER":1,
#  "print": 1,

  "dsrg_s":0.5,

  "e_convergence"       :  1e-10 , # energy convergence of the FCI iterations
  "r_convergence":         1e-8 , # residual convergence of the FCI iterations
  "mcscf_e_convergence":  1e-10  ,# energy convergence of the MCSCF iterations
  "mcscf_g_convergence":  1e-6 , # gradient convergence of the MCSCF iterations
  "mcscf_micro_maxiter":  4  ,# do at least 4 micro iterations per macro iteration
}
import pdb
pdb.set_trace()
e_ct_scf,ct_mcscf_wfn=psi4.energy("forte",forte_options=forte_options,return_wfn=True)
