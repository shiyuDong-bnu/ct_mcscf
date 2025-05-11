import sys
import psi4
import numpy as np
sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/software/forte")
from ct.mrct import canonical_transform
from ct.rhf_energy import rhf_energy
from utils.write_dump import write_dump_np

psi4.set_output_file("bare.out")

psi_mol=psi4.geometry(
    """
0 3
C
H 1 r_ch
H 1 r_ch 2 ahch
unit bohr
symmetry c1"""
)
psi_mol.r_ch=2.0413
psi_mol.ahch=134.22

GAMMA = 1.0 
BASIS_SET="cc-pVDZ-F12"
FREEZE_CORE=True

psi4.set_options(
    {
        "basis":BASIS_SET,
        "reference":"rohf",
        'df_basis_mp2': BASIS_SET+"-Optri",
        "scf_type":"pk",
        'screening': 'csam',
        "maxiter":300,
        "e_convergence":10,
        "d_convergence":8,
        # # "RESTRICTED_DOCC":[1,0,0,0],
        # "ACTIVE":[3,0,1,2],
    }
)

forte_mcscf_options={
    "active_space_solver":"fci",
    "active":[6],
  "frozen_docc":[1],
  "MCSCF_IGNORE_FROZEN_ORBS":False,
  "correlation_solver":"dsrg-mrpt2",
  "dsrg_s":0.5,
  "e_convergence"       :  1e-10 , # energy convergence of the FCI iterations
  "r_convergence":         1e-8 , # residual convergence of the FCI iterations
  "mcscf_e_convergence":  1e-10  ,# energy convergence of the MCSCF iterations
  "mcscf_g_convergence":  1e-6 , # gradient convergence of the MCSCF iterations
    
}

## do mcscf calculation to generate 1RDM and 2RDM (density matrix)
e_hf,hf_wfn=psi4.energy("scf",return_wfn=True)
e_mcscf,mcscf_wfn=psi4.energy("forte",forte_options=forte_mcscf_options,
        ref_wfn=hf_wfn,return_wfn=True)
e_casscf=psi4.core.variable('DSRG REFERENCE ENERGY')
e_dsrg_pt=psi4.core.variable('CURRENT ENERGY')
print("mcscf energy",e_casscf)
print("pt2 energy",e_dsrg_pt)
print("hf energy",e_hf)
