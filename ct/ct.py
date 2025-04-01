import psi4
import numpy as np
from ct.get_cabs import get_cabs
from ct.utils.orbital_space import OrbitalSpace
from ct.get_int import get_eri_ri_ri_int,get_hcore_int,get_density,get_fock
from ct.get_f12_int import get_f12,gen_V,get_fock_ri,gen_b,rational_generate,conjugate
from ct.get_hbar import get_hbar
def canonical_transform(mol,wfn,basis,df_basis,gamma,freeze_core):

    obs,ribs,cabs=get_cabs(mol,wfn,basis,df_basis)
    my_orbital_space=OrbitalSpace(wfn,obs,ribs,cabs)
    g=get_eri_ri_ri_int(my_orbital_space)
    D1,D2=get_density(my_orbital_space)
    h=get_hcore_int(my_orbital_space)
    f=get_fock(my_orbital_space,h,D1,g)
    G=get_f12(gamma,my_orbital_space)
    V_noper,X_noper=gen_V(gamma,my_orbital_space)
    fock_ri_mo,K_ri_mo,total_fock,f_virtual_cabs=get_fock_ri(my_orbital_space)
    B_final_temp=gen_b(gamma,my_orbital_space,total_fock,fock_ri_mo,K_ri_mo)
    if frezee_core:
        G[:,:,:,0]=G[:,:,0,:]=0
        V_noper[0,:,:,:]=V_noper[:,0,:,:]=0
        X_noper[0,:,:,:]=X_noper[:,0,:,:]=X_noper[:,:,0,:]=X_noper[:,:,:,0]=0
        B_final_temp[0,:,:,:]=B_final_temp[:,0,:,:]=B_final_temp[:,:,0,:]=B_final_temp[:,:,:,0]=0
    V_rational=rational_generate(np.einsum("ijkl->klij",V_noper))

    X_rational_temp=rational_generate(X_noper)
    X_rational=conjugate(rational_generate(conjugate(X_rational_temp)))

    B_rational_temp=rational_generate(B_final_temp)
    B_rational=conjugate(rational_generate(conjugate(B_rational_temp)))

    hbar,gbar=get_hbar(my_orbital_space,V_rational,X_rational,B_rational,D1,D2,g,G,f,h)
    Cp=my_orbital_space.Cp
    mints=psi4.core.MintsHelper(my_orbital_space.bs_obs())
    Cinv = Cp.T @ mints.ao_overlap()
    Hct = {
        "Hbar1": np.einsum("pm,qn,pq->mn", Cinv, Cinv, hbar, optimize=True),
        "Hbar2": np.einsum("pm,qn,rg,sh,pqrs->mngh", Cinv, Cinv, Cinv, Cinv, gbar, optimize=True)
    }
    return Hct

