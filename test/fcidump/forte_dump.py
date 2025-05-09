import psi4
import numpy as np
import sys
sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/software/forte")
from utils.write_dump import  write_dump_np
psi4.set_output_file("fcidump.out")
psi_mol=psi4.geometry(
"""
0 1
C
H 1 r_ch
H 1 r_ch 2 ahch
symmetry c1
unit bohr
"""
)
psi_mol.r_ch=2.1023
psi_mol.ahch=101.71
psi4.set_options(
    {
        "basis":"cc-pVDZ",
        "reference":"rhf",
        "scf_type":"pk",
        "maxiter":300,
        "e_convergence":10,
        "d_convergence":8,
    }
)
forte_options={
    "job_type":"mcscf_two_step",
    "int_type":"fcidump",
    "fcidump_file":"TEST.DUMP",
    "active_space_solver":"fci",
    "restricted_docc":[1],
    "active":[6],
}
e_mcscf,mcscf_wfn=psi4.energy("forte",forte_options=forte_options,return_wfn=True)
