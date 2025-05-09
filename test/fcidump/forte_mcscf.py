import psi4
import numpy as np
import sys
sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/software/forte")
from utils.write_dump import  write_dump_np
psi4.set_output_file("methylene.out")
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
    "active_space_solver":"fci",
    "restricted_docc":[1],
    "active":[6],
}
e_mcscf,mcscf_wfn=psi4.energy("forte",forte_options=forte_options,return_wfn=True)
## here we  save fci dump 
n_ele=mcscf_wfn.nalpha()*2
n_obs=mcscf_wfn.nmo()
V_nuc=psi_mol.nuclear_repulsion_energy()
cp=mcscf_wfn.Ca().np
h1e=mcscf_wfn.H().np
h2e=psi4.core.MintsHelper(mcscf_wfn).ao_eri().np
h1e_ct_mo=np.einsum("ij,iI,jJ->IJ",h1e,cp,cp,optimize=True)
h2e_ct_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",h2e,cp,cp,cp,cp,optimize=True)
write_dump_np("TEST.DUMP",n_ele=n_ele,n_obs=n_obs,V_nuc=V_nuc,mo_h_core=h1e_ct_mo,mo_eri=h2e_ct_mo)
