"""
This script is used to calculate  
Ne with active space set to 1 slater determinant
"""

import sys
sys.path.append("/home/sydong/work/workspace/reproducing/ct_mcscf/")
sys.path.insert(0,"/home/sydong/work/workspace/reproducing/forte")

import numpy as np
import psi4
import forte
print(forte.__file__)
from ct.mrct import canonical_transform
from ct.rhf_energy import rhf_energy
from write_dump import write_dump_np
psi4.set_output_file("Ne_mr.out")
psi4.set_num_threads(36)
psi4.set_memory("300GB")
psi_mol = psi4.geometry(
    """
0 1
Ne 0.0 0.0 0.0
Ne 0.0 0.0 100.0
unit angstrom 
symmetry c1
"""
)
GAMMA=1.5
PSI4_OPTIONS = {
    "basis": "aug-cc-pVDZ",  ## please specify basis manually
    'df_basis_mp2': "aug-cc-pVDZ"+"-Optri",
    "reference": "rhf",
    "scf_type": "pk",
    "screening": "csam",
    "maxiter": 300,
    "e_convergence": 10,
    "d_convergence": 8,
    "freeze_core":True,
}
FORTE_OPTIONS = {
    "job_type": "mcscf_two_step",
    "active_space_solver": "fci",
    "restricted_docc": [4],
    "active": [8],
    "ci_spin_adapt": True,
    #    "correlation_solver": "dsrg-mrpt2",
    #    "dsrg_s": 0.5,
    "e_convergence": 1e-10,  # energy convergence of the FCI iterations
    "r_convergence": 1e-8,  # residual convergence of the FCI iterations
    "mcscf_e_convergence": 1e-10,  # energy convergence of the MCSCF iterations
    "mcscf_g_convergence": 1e-10,  # gradient convergence of the MCSCF iterations
    "subspace": ["Ne(2p,3s)"],
    "avas": True,
    "avas_num_active": 8,
}
E_HF = []
E_CASSCF = []
E_DSRG_PT = []
psi4.set_options(PSI4_OPTIONS)


def calculate(basis):
    """
    function to do
    hartree fock
    mcscf
    dsrg-pt2 energy
    return all this energy into tuple
    """
    psi4.core.set_global_option("basis", basis)
    e_hf,wfn_hf=psi4.energy("scf",return_wfn=True)
    e_cas, mcscf_wfn = psi4.energy(
        "forte", forte_options=FORTE_OPTIONS, return_wfn=True
    )

    print(basis)

    basis = psi4.core.get_global_option("BASIS")
    df_basis = psi4.core.get_global_option("DF_BASIS_MP2")
    mr_info = {}
    mr_info["occupied_index"] = slice(0, 4)
    mr_info["active_index"] = slice(4, 12)
    mr_info["RDM1"] = psi4.core.variable("CURRENT RDM1").np
    mr_info["RDM2"] = psi4.core.variable("CURRENT RDM2").np.reshape((8,) * 4)
    mr_info=None
    h_ct = canonical_transform(
        psi_mol,
        wfn_hf,
        basis,
        df_basis,
        gamma=GAMMA,
        freeze_core=True,
        mr_info=mr_info,
    )
    def convert_ct_to_quccsd(h_ct, hf_ct):
        h1e_ct = np.copy(h_ct["Hbar1"])
        h2e_ct = np.copy(h_ct["Hbar2"])
        h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)  # to ao basis
        cp_ct = hf_ct["C"]
        return h1e_ct, h2e_ct, cp_ct

    n_ele = mcscf_wfn.nalpha() * 2
    n_obs = mcscf_wfn.nmo()
    V_nuc = psi_mol.nuclear_repulsion_energy()
    rhf_ct = rhf_energy(psi_mol, mcscf_wfn, h_ct)
    print(
        "ct  hf energy is ",
        rhf_ct["escf"],
        " correlation energy is ",
        rhf_ct["escf"] - e_hf,
    )
    h1e_ct, h2e_ct, cp_ct = convert_ct_to_quccsd(h_ct, rhf_ct)
    h1e_ct_mo = np.einsum("ij,iI,jJ->IJ", h1e_ct, cp_ct, cp_ct, optimize=True)
    h2e_ct_mo = np.einsum(
        "ijkl,iI,jJ,kK,lL->IJKL", h2e_ct, cp_ct, cp_ct, cp_ct, cp_ct, optimize=True
    )
    write_dump_np(
        basis + "MR_SIG.DUMP",
        n_ele=n_ele,
        n_obs=n_obs,
        V_nuc=V_nuc,
        mo_h_core=h1e_ct_mo,
        mo_eri=h2e_ct_mo,
    )
    psi4.set_options({
    "screening": "csam",
    })
    forte_options = {
        "job_type":"mcscf_two_step",
        "active_space_solver": "fci",

        "restricted_docc": [4],
        "active": [ 8],


        "int_type": "fcidump",
        "fcidump_file": basis+ "MR_SIG.DUMP" ,
 #       "correlation_solver": "dsrg-mrpt2",
 #       "corr_level":"pt2",
 #       "dsrg_s": 0.5,


        "subspace": ["Ne(2p,3s)"],
         "avas": True,
         "avas_num_active": 8,
          
        "e_convergence": 1e-10,  # energy convergence of the FCI iterations
        "r_convergence": 1e-8,  # residual convergence of the FCI iterations
        "mcscf_e_convergence": 1e-10,  # energy convergence of the MCSCF iterations
        "mcscf_g_convergence": 1e-6,  # gradient convergence of the MCSCF iterations
        "mcscf_micro_maxiter": 4,  # do at least 4 micro iterations per macro iteration
    }
    e_ct_scf, ct_mcscf_wfn = psi4.energy(
        "forte", forte_options=forte_options,  return_wfn=True
    )

    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.clean()
    return None


calculate(
        basis="aug-cc-pVDZ",
    )
