import psi4
import numpy as np
from ct.get_cabs import get_cabs
from ct.utils.orbital_space import OrbitalSpace
from ct.get_int import get_eri_ri_ri_int,get_hcore_int,get_density,get_fock
from ct.get_f12_int import get_f12,gen_V,gen_b,rational_generate,conjugate
from ct.get_hbar import get_hbar
from ct.utils.eri_interface import SlicedERI
def canonical_transform(mol,wfn,basis,df_basis,gamma,freeze_core):

    obs,ribs,cabs=get_cabs(mol,wfn,basis,df_basis)
    my_orbital_space=OrbitalSpace(wfn,obs,ribs,cabs)
    sliced_g=SlicedERI(my_orbital_space)
    D1,D2=get_density(my_orbital_space)
    h=get_hcore_int(my_orbital_space)
    f_only_j,k,f,f_virtual_cabs_int=get_fock(my_orbital_space,h,D1,sliced_g.format_g_for_fock())
    G=get_f12(gamma,my_orbital_space)
    V_noper,X_noper=gen_V(gamma,my_orbital_space)
    fock_ri_mo,K_ri_mo,total_fock,f_virtual_cabs=f_only_j,k,f,f_virtual_cabs_int
    B_final_temp=gen_b(gamma,my_orbital_space,total_fock,fock_ri_mo,K_ri_mo)
    if freeze_core:
        nfrzc=wfn.nfrzc()
        try :
            assert nfrzc>0
            print("number of freeze core orbital  is ",nfrzc)
        except:
            raise("""Freeze core is set true, but get zero number of freeze core orbital \n
            you should also set freeze core option in psi4 ,which is where this 
            code get number fo freeze core orbital""")
        G[:,:,:,:nfrzc]=G[:,:,:nfrzc,:]=0
        V_noper[:nfrzc,:,:,:]=V_noper[:,:nfrzc,:,:]=0
        X_noper[:nfrzc,:,:,:]=X_noper[:,:nfrzc,:,:]=X_noper[:,:,:nfrzc,:]=X_noper[:,:,:,:nfrzc]=0
        B_final_temp[:nfrzc,:,:,:]=B_final_temp[:,:nfrzc,:,:]=B_final_temp[:,:,:nfrzc,:]=B_final_temp[:,:,:,:nfrzc]=0
    V_rational=rational_generate(np.einsum("ijkl->klij",V_noper))

    X_rational_temp=rational_generate(X_noper)
    X_rational=conjugate(rational_generate(conjugate(X_rational_temp)))

    B_rational_temp=rational_generate(B_final_temp)
    B_rational=conjugate(rational_generate(conjugate(B_rational_temp)))
    hbar,gbar=get_hbar(my_orbital_space,V_rational,X_rational,B_rational,D1,D2,sliced_g,G,f,h)
    Cp=my_orbital_space.Cp
    mints=psi4.core.MintsHelper(my_orbital_space.bs_obs())
    Cinv = Cp.T @ mints.ao_overlap()
    Hct = {
        "Hbar1": np.einsum("pm,qn,pq->mn", Cinv, Cinv, hbar, optimize=True),
        "Hbar2": np.einsum("pm,qn,rg,sh,pqrs->mngh", Cinv, Cinv, Cinv, Cinv, gbar, optimize=True)
    }
    return Hct

