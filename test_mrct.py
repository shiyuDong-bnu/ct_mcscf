import numpy as np
from ct.mrct import canonical_transform
from ct.rhf_energy import rhf_energy
from write_dump import write_dump_np

import psi4
psi4.set_output_file("methylene.out")
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
basis_sets=["cc-pVDZ-F12","cc-pVTZ-F12","cc-pVQZ-F12",]
E_HF=[]
E_CT_HF=[]
E_CASSCF=[]
E_CT_CASSCF=[]
E_CT_DSRG_PT=[]
for bs in basis_sets:
    psi4.set_options(
        {
            "basis":bs,
            "reference":"rhf",
            'df_basis_mp2': bs+"-Optri",
            "scf_type":"pk",
            'screening': 'csam',
            "maxiter":300,
            "e_convergence":10,
            "d_convergence":8,
            # # "RESTRICTED_DOCC":[1,0,0,0],
            # "ACTIVE":[3,0,1,2],
        }
    )
    forte_options={
        "job_type":"mcscf_two_step",
        "active_space_solver":"fci",
        "restricted_docc":[1],
        "active":[6],
    }
    e_hf,hf_wfn=psi4.energy("scf",return_wfn=True)
    e_mcscf,mcscf_wfn=psi4.energy("forte",forte_options=forte_options,ref_wfn=hf_wfn,return_wfn=True)
    
    basis = psi4.core.get_global_option('BASIS')
    df_basis = psi4.core.get_global_option('DF_BASIS_MP2')
    mr_info={}
    mr_info['occupied_index']=slice(0,1)
    mr_info['active_index']=slice(1,7)
    mr_info['RDM1']=psi4.core.variable("CURRENT RDM1").np
    mr_info['RDM2']=psi4.core.variable("CURRENT RDM2").np.reshape((6,)*4)
    h_ct = canonical_transform(
        psi_mol, mcscf_wfn, basis, df_basis, gamma=GAMMA, frezee_core=False,mr_info=mr_info)
    
    def convert_ct_to_quccsd(h_ct, hf_ct):
        h1e_ct = np.copy(h_ct['Hbar1'])
        h2e_ct = np.copy(h_ct['Hbar2'])
        h2e_ct = np.einsum("ijkl->ikjl", h2e_ct)  # to ao basis
        cp_ct = hf_ct['C']
        return h1e_ct, h2e_ct, cp_ct
    n_ele=mcscf_wfn.nalpha()*2
    n_obs=mcscf_wfn.nmo()
    V_nuc=psi_mol.nuclear_repulsion_energy()
    rhf_ct = rhf_energy(psi_mol, mcscf_wfn, h_ct)
    print("ct  hf energy is ", rhf_ct["escf"],
            " correlation energy is ", rhf_ct["escf"]-e_hf)
    h1e_ct, h2e_ct, cp_ct=convert_ct_to_quccsd(h_ct, rhf_ct)
    h1e_ct_mo=np.einsum("ij,iI,jJ->IJ",h1e_ct,cp_ct,cp_ct,optimize=True)
    h2e_ct_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",h2e_ct,cp_ct,cp_ct,cp_ct,cp_ct,optimize=True)
    write_dump_np("MRDRESSED.DUMP",n_ele=n_ele,n_obs=n_obs,V_nuc=V_nuc,mo_h_core=h1e_ct_mo,mo_eri=h2e_ct_mo)
    
    forte_options={
    #  "job_type" :              "mcscf_two_step",
      "active_space_solver":                "fci",
      "restricted_docc":       [1],
      "active"         :       [6,],
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
    e_ct_scf,ct_mcscf_wfn=psi4.energy("forte",forte_options=forte_options,ref_wfn=hf_wfn,return_wfn=True)
    e_ct_casscf=psi4.core.variable('DSRG REFERENCE ENERGY')
    e_dsrg_pt=psi4.core.variable('CURRENT ENERGY')
    E_HF.append(e_hf)
    E_CT_HF.append(rhf_ct["escf"])
    E_CASSCF.append(e_mcscf)
    E_CT_CASSCF.append(e_ct_casscf)
    E_CT_DSRG_PT.append(e_dsrg_pt)
    print(bs)
    print("hf energy",e_hf)
    print("mcscf energy",e_mcscf)
    print("cthf energy",rhf_ct["escf"])
    print("ct_mcscf energy",e_ct_casscf)
    print("ct_pt2 energy",e_dsrg_pt)
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.clean()
data={"hf":E_HF,
      "ct_hf":E_CT_HF,
      "mcscf":E_CASSCF,
      "ct_mcscf":E_CT_CASSCF,
      "ct_dsrg_pt":E_CT_DSRG_PT,
      }
np.save("ct_data",data,allow_pickle=True)
