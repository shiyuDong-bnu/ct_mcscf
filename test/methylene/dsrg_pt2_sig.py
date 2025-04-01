import psi4
import numpy as np
psi4.set_output_file("methylene.out")
psi_mol=psi4.geometry(
    """
0 1
C
H 1 r_ch
H 1 r_ch 2 ahch
unit bohr"""
)
psi_mol.r_ch=2.1023
psi_mol.ahch=101.71
basis_sets=["cc-pVDZ","cc-pVTZ","cc-pVQZ","cc-pV5Z","cc-pV6Z"]
E_CASSCF=[]
E_DSRG_PT=[]
for  bs in basis_sets:
    psi4.set_options(
        {
           # "basis":"cc-pVDZ-F12",
            "basis":bs,
            "reference":"rhf",
            "scf_type":"pk",
            "maxiter":300,
            "e_convergence":10,
            "d_convergence":8,
            # # "RESTRICTED_DOCC":[1,0,0,0],
            # "ACTIVE":[3,0,1,2],
        }
    )
    forte_options={
     #   "job_type":"mcscf_two_step",
        "active_space_solver":"fci",
        "restricted_docc":[1,0,0,0],
        "active":[3,0,1,2],
        "correlation_solver":"dsrg-mrpt2",
        "dsrg_s":0.5,

    }
    psi4.energy("forte",forte_options=forte_options)
    print("basis set",bs)
    e_casscf=psi4.core.variable('DSRG REFERENCE ENERGY')
    e_dsrg_pt=psi4.core.variable('CURRENT ENERGY')
    print("casscf",e_casscf)
    print("dsrgpt2",e_dsrg_pt)
    E_CASSCF.append(e_casscf)
    E_DSRG_PT.append(e_dsrg_pt)
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.clean()
print(E_CASSCF)
print(E_DSRG_PT)
np.save("a1_casscf",E_CASSCF)
np.save("a1_dsrg_pt",E_DSRG_PT)
