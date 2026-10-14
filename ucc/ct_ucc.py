from pyscf import gto,scf,cc,ao2mo
from QC5 import (
    genAn,
    get_qubit_hamiltonian,
    get_ao2so_h1_h2_int,
    RuCCSD,
    get_scipy_optimized_vqe,
)
import os
import numpy as np
bond_length=np.concatenate([np.linspace(0.4,1.0,7),
                            np.linspace(1.2,2.0,5)])
def pyscf_gen_mol(r,basis):
    mol=gto.Mole()
    mol.atom=genAn("H",2,r)
    mol.basis=basis
    mol.charge=0
    mol.spin=0
    mol.symmetry=True
    mol.build()
    return mol

def get_hamiltonian_from_pyscf(mol,mf_pyscf):
    nelec=mol.nelec
    norb=len(mf_pyscf.get_occ())
    enuc=mol.energy_nuc()
    h1e=mf_pyscf.get_hcore()
    h2e=mf_pyscf._eri
    mo_coeff=mf_pyscf.mo_coeff
    return nelec,norb,enuc,h1e,h2e,mo_coeff
def uccsd(enuc,h1,h2,nelec,t1,t2,fermion_transform="jordan_wigner"):
    qubit_ham=get_qubit_hamiltonian(enuc,h1,h2,fermion_transform)
    qubit_ham.compress()
    UCCSD=RuCCSD(nelec,t1,t2,fermion_transform)
    UCCSD.parameter=1.0
    cir=UCCSD.params_circuit
    initial=UCCSD.get_ccsd_initial_amps()
    vqe=get_scipy_optimized_vqe(
        qubit_ham,
        cir,
        initial,
        printdetails=True
    )
    return vqe
HF=[]
CCSD=[]
UCCSD=[]
CTUCCSD=[]
for i in bond_length:
    print(i)
    mol=pyscf_gen_mol(i,"cc-pvdz")
    mf=scf.RHF(mol)
    e_rhf=mf.kernel()

    (nelec_pyscf,norb_pyscf,enuc_pyscf,
     h1e_pyscf,h2e_pyscf,mo_coeff_pyscf)=get_hamiltonian_from_pyscf(mol,mf)
    h1e_pyscf,h2e_pyscf=get_ao2so_h1_h2_int(
        mo_coeff_pyscf,h1e_pyscf,h2e_pyscf
    )
    ccsd=cc.CCSD(mf)
    cc_result=ccsd.kernel()
    ccsd_total=e_rhf+cc_result[0]
    t1,t2=cc_result[1],cc_result[2]
    ## uccsd
    quccsd=uccsd(
        enuc_pyscf,
        h1e_pyscf,
        h2e_pyscf,
        nelec_pyscf,t1,t2,fermion_transform="jordan_wigner",
    )
    ## load ct
    h1e_ct=np.load(f"h1ct{i:.2f}.npy")
    h2e_ct=np.load(f"h2ct{i:.2f}.npy")
    moc_ct=np.load(f"c{i:.2f}.npy")
    ct_h1,ct_h2=get_ao2so_h1_h2_int(moc_ct,h1e_ct,h2e_ct)
    ctquccsd=uccsd(
        enuc_pyscf,
        ct_h1,
        ct_h2,
        nelec_pyscf,t1,t2,fermion_transform="jordan_wigner",
    )
    HF.append(e_rhf)
    CCSD.append(ccsd_total)
    UCCSD.append(quccsd[0])
    CTUCCSD.append(ctquccsd[0])
np.savez("h2_ucc_6_31g.npz",{
    "hf":HF,
    "ccsd":CCSD,
    "uccsd":UCCSD,
    "ctuccsd":CTUCCSD,
})
