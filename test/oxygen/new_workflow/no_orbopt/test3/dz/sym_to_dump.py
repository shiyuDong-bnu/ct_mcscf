import sys
import psi4
import numpy as np

sys.path.append("/data/home/sydong/work/ct_mcscf/")
sys.path.append("/data/home/sydong/work/reproducing/forte")
from ct.mrct import canonical_transform
from ct.rhf_energy import rhf_energy
from utils.write_dump import write_dump_np

psi4.set_output_file("gen_sym.out")

psi_mol = psi4.geometry(
    """
0 3
O
O 1 r_oo
unit angstrom
"""
)
psi_mol.r_oo = 1.0 

GAMMA = 1.0
BASIS_SET = "cc-pVDZ-F12"
FREEZE_CORE = True

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
   # "job_type": "mcscf_two_step",
    "active_space_solver": "fci",
    "active":          [1,0,1,1,0,1,1,1 ],         ## 2p
    "restricted_docc": [1,0,0,0,0,1,0,0 ],## 2s
    "frozen_docc":     [1,0,0,0,0,1,0,0 ],  ## 1s
    "MCSCF_IGNORE_FROZEN_ORBS": False,

    "avg_state": [[0,1,2],[1,1,1],[1,3,3],[2,1,1],[3,1,1],[4,1,1],[5,1,1],[6,1,1],[7,1,1]],
    "avg_weight":[[0.5,0.0],[0.5,],[1.0,0.0,0.0],[0],[0],[0],[0],[0],[0],],
    "ci_spin_adapt": True,

    "subspace": ["O(2p)"],
    "avas": True,
    "avas_num_active": 6,

  "correlation_solver":"dsrg-mrpt2",
  "dsrg_s":0.5,
  "dsrg_multi_state":   "sa_full",
  "calc_type":       "sa",
}

## do mcscf calculation to generate 1RDM and 2RDM (density matrix)
e_hf, hf_wfn = psi4.energy("scf", return_wfn=True)
psi4.molden(hf_wfn,"hf.molden")
e_mcscf, mcscf_wfn = psi4.energy(
    "forte", forte_options=forte_mcscf_options, ref_wfn=hf_wfn, return_wfn=True
)
### direct do ct 
#
#basis = psi4.core.get_global_option("BASIS")
#df_basis = psi4.core.get_global_option("DF_BASIS_MP2")
#mr_info = {}
#mr_info["occupied_index"] = slice(0, 1)
#mr_info["active_index"] = slice(1, 7)
#mr_info["RDM1"] = psi4.core.variable("CURRENT RDM1").np
#mr_info["RDM2"] = psi4.core.variable("CURRENT RDM2").np.reshape((6,) * 4)
### do ct to get integrals
#h_ct = canonical_transform(
#    psi_mol,
#    mcscf_wfn,
#    basis,
#    df_basis,
#    gamma=GAMMA,
#    freeze_core=FREEZE_CORE,
#    mr_info=mr_info,
#)
### do ct hf to get MO coeff
#rhf_ct = rhf_energy(psi_mol, mcscf_wfn, h_ct)
#print(
#    "ct  hf energy is ",
#    rhf_ct["escf"],
#    " correlation energy is ",
#    rhf_ct["escf"] - e_hf,
#)
#
### dump ct hamiltonian
#
#n_ele = mcscf_wfn.nalpha()+mcscf_wfn.nbeta() 
#n_obs = mcscf_wfn.nmo()
#ms2 = mcscf_wfn.nalpha() - mcscf_wfn.nbeta()
#V_nuc = psi_mol.nuclear_repulsion_energy()
#print("Total electron", n_ele)
#print("Total orbital", n_obs)
#print("MS2 ", ms2)
#h1e_ct = h_ct["Hbar1"]
#h2e_ct = h_ct["Hbar2"]  ## the returned ct 2eri is physical
#h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)
#Ca = mcscf_wfn.Ca_subset("SO", "ALL")
#ao2so = mcscf_wfn.aotoso()
#epsilon=mcscf_wfn.epsilon_a()
#Cao2mo=[]
#epsilon_all_irrep=[]
#for hi in range(mcscf_wfn.nirrep()):
#    Ci = np.matmul(ao2so.nph[hi], Ca.nph[hi])
#    epsilon_all_irrep.extend(epsilon.nph[hi])
#    Cao2mo.append(Ci)
### reorder
#index=np.argsort(epsilon_all_irrep)
#C_sort=np.hstack(Cao2mo)
#C_final=C_sort[:,index]
#cp_ct = C_final
#h1e_ct_mo = np.einsum("ij,iI,jJ->IJ", h1e_ct, cp_ct, cp_ct, optimize=True)
#h2e_ct_mo = np.einsum(
#    "ijkl,iI,jJ,kK,lL->IJKL", h2e_ct, cp_ct, cp_ct, cp_ct, cp_ct, optimize=True
#)
#write_dump_np(
#    "MRDRESSED.DUMP",
#    n_ele=n_ele,
#    n_obs=n_obs,
#    V_nuc=V_nuc,
#    ms2=ms2,
#    mo_h_core=h1e_ct_mo,
#    mo_eri=h2e_ct_mo,
#)
#exit()
## 
## save molecule orbital 
mcscf_wfn.to_file("sym_mcscf.npy")
Ca = mcscf_wfn.Ca_subset("SO", "ALL")
ao2so = mcscf_wfn.aotoso()
epsilon=mcscf_wfn.epsilon_a()
Cao2mo=[]
epsilon_all_irrep=[]
for hi in range(mcscf_wfn.nirrep()):
    Ci = np.matmul(ao2so.nph[hi], Ca.nph[hi])
    epsilon_all_irrep.extend(epsilon.nph[hi])
    Cao2mo.append(Ci)
## reorder
index=np.argsort(epsilon_all_irrep)
C_sort=np.hstack(Cao2mo)
C_final=C_sort[:,index]
np.save("sym_coeff.npy",C_final)
psi4.molden(mcscf_wfn,"active.molden")
print("hf energy", e_hf)
print("mcscf energy", e_mcscf)
