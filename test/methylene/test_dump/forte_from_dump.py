import sys
import psi4
import numpy as np

sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/software/forte")

from ct.mrct import canonical_transform
from ct.rhf_energy import rhf_energy
from utils.write_dump import write_dump_np

psi4.set_output_file("dressed.out")
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
  "active_space_solver":                "detci",
  "active"         :       [6,],
  "frozen_docc":[1],
  "MCSCF_IGNORE_FROZEN_ORBS":False,
  "int_type"         :     "fcidump",
  "fcidump_file"      :   "MRDRESSED.DUMP",
  "correlation_solver":"dsrg-mrpt2",
  "dsrg_s":0.5,
  "e_convergence"       :  1e-10 , # energy convergence of the FCI iterations
  "r_convergence":         1e-8 , # residual convergence of the FCI iterations
  "mcscf_e_convergence":  1e-10  ,# energy convergence of the MCSCF iterations
  "mcscf_g_convergence":  1e-6 , # gradient convergence of the MCSCF iterations
  "mcscf_micro_maxiter":  4  ,# do at least 4 micro iterations per macro iteration
}
e_ct_scf,ct_mcscf_wfn=psi4.energy("forte",forte_options=forte_options,return_wfn=True)
e_ct_casscf=psi4.core.variable('DSRG REFERENCE ENERGY')
e_dsrg_pt=psi4.core.variable('CURRENT ENERGY')
print("ct_mcscf energy",e_ct_casscf)
print("ct_pt2 energy",e_dsrg_pt)
