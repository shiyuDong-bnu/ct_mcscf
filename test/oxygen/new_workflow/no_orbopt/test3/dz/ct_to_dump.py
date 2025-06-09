import sys
import psi4
import numpy as np

sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/work/reproducing/forte")
from ct.mrct import canonical_transform
from ct.rohf_energy import rohf_energy
from utils.write_dump import write_dump_np

psi4.set_output_file("gen_dump.out")

psi_mol = psi4.geometry(
    """
0 3
O
O 1 r_oo
unit angstrom
symmetry c1"""
)
psi_mol.r_oo = 1.0

GAMMA = 1.0
BASIS_SET = "cc-pVDZ-F12"
FREEZE_CORE = True

N_FROZEN=2
N_RESTRICTED_DOCC=2
N_ACTIVE=6
N_OCC=N_FROZEN+N_RESTRICTED_DOCC

psi4.set_options(
    {
        "basis": BASIS_SET,
        "reference": "rohf",
        "df_basis_mp2": BASIS_SET + "-Optri",
        "scf_type": "pk",
        "screening": "csam",
        "maxiter": 300,
        "e_convergence": 10,
        "d_convergence": 8,
        "freeze_core": True,
        # # "RESTRICTED_DOCC":[1,0,0,0],
        # "ACTIVE":[3,0,1,2],
    }
)

forte_mcscf_options = {
    "job_type": "mcscf_two_step",
    "active_space_solver": "fci",
    "active": [N_ACTIVE ],         ## 2p
    "restricted_docc":[N_RESTRICTED_DOCC], ## 2s
    "frozen_docc":[N_FROZEN],       ## 1s
    "MCSCF_IGNORE_FROZEN_ORBS": False,
#    "MCSCF_NO_ORBOPT":True,
#    "MCSCF_MAXITER":1,
#    "MCSCF_MICRO_MAXITER":0,
    "DL_MAXITER":200,
    "DL_GUESS_PER_ROOT":3,
    "avg_state": [[0,1,9],[0,3,3]],
    "avg_weight":[[0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],[1.0,0.0,0.0,]],
    "ci_spin_adapt": True,

}

## do mcscf calculation to generate 1RDM and 2RDM (density matrix)
e_hf, hf_wfn = psi4.energy("scf", return_wfn=True)
## load coeff from symmtry mcscf orbital
C_restart=np.load("sym_coeff.npy")
my_file="c1hf.npy"
hf_wfn.to_file(my_file)
hf_wfn_data=np.load("c1hf.npy",allow_pickle=True).item()
hf_wfn_data['matrix']['Ca']=C_restart
hf_wfn_data['matrix']['Cb']=C_restart
for key in hf_wfn_data['matrix'].keys():
    if key=='Ca' or key=='Cb':
        pass
    else:
        hf_wfn_data['matrix'][key]=None
np.save("c1hf.npy",hf_wfn_data,allow_pickle=True)
restart_wfn=psi4.core.Wavefunction.from_file("c1hf.npy")
psi4.molden(restart_wfn,"restart.molden")
###
#import pdb
#pdb.set_trace()
e_mcscf, mcscf_wfn = psi4.energy(
    "forte", forte_options=forte_mcscf_options, ref_wfn=hf_wfn, return_wfn=True
)
psi4.molden(mcscf_wfn,"active.molden")
print("hf energy", e_hf)
print("mcscf energy", e_mcscf)

basis = psi4.core.get_global_option("BASIS")
df_basis = psi4.core.get_global_option("DF_BASIS_MP2")
mr_info = {}
mr_info["occupied_index"] = slice(0, N_OCC)
mr_info["active_index"] = slice(N_OCC,N_OCC+N_ACTIVE )
mr_info["RDM1"] = psi4.core.variable("CURRENT RDM1").np
mr_info["RDM2"] = psi4.core.variable("CURRENT RDM2").np.reshape((N_ACTIVE,) * 4)
## do ct to get integrals
h_ct = canonical_transform(
    psi_mol,
    mcscf_wfn,
    basis,
    df_basis,
    gamma=GAMMA,
    freeze_core=FREEZE_CORE,
    mr_info=mr_info,
)
## do ct hf to get MO coeff
rhf_ct = rohf_energy(psi_mol, mcscf_wfn, h_ct)
print(
    "ct  hf energy is ",
    rhf_ct["escf"],
    " correlation energy is ",
    rhf_ct["escf"] - e_hf,
)

## dump ct hamiltonian

n_ele = mcscf_wfn.nalpha()+mcscf_wfn.nbeta() 
n_obs = mcscf_wfn.nmo()
ms2 = mcscf_wfn.nalpha() - mcscf_wfn.nbeta()
V_nuc = psi_mol.nuclear_repulsion_energy()
print("Total electron", n_ele)
print("Total orbital", n_obs)
print("MS2 ", ms2)
h1e_ct = h_ct["Hbar1"]
h2e_ct = h_ct["Hbar2"]  ## the returned ct 2eri is physical
h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)
cp_ct = mcscf_wfn.Ca()
h1e_ct_mo = np.einsum("ij,iI,jJ->IJ", h1e_ct, cp_ct, cp_ct, optimize=True)
h2e_ct_mo = np.einsum(
    "ijkl,iI,jJ,kK,lL->IJKL", h2e_ct, cp_ct, cp_ct, cp_ct, cp_ct, optimize=True
)
write_dump_np(
    "MRDRESSED.DUMP",
    n_ele=n_ele,
    n_obs=n_obs,
    V_nuc=V_nuc,
    ms2=ms2,
    mo_h_core=h1e_ct_mo,
    mo_eri=h2e_ct_mo,
)
