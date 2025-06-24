"""
This moudle have three function to generate f12 related integral
1. get_f12 create G 
2. gen_V create v and x
3. get_b create b 
"""
import psi4
import numpy as np
from ct.utils.timer import timer_decorator
import sys
import time
# G = 3/8 <αβ|Q12 F12|ij> + 1/8 <αβ|Q12 F12|ji>
@timer_decorator
def get_f12(my_orbital_space,f12_int,gamma):
    nri=my_orbital_space.nri
    bs_obs=my_orbital_space.bs_obs()
    bs_cabs=my_orbital_space.bs_cabs()
    Cp=my_orbital_space.Cp
    Cx=my_orbital_space.Cx
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v
    no=o.stop
    ## calculation begin here
    # mints=psi4.core.MintsHelper(bs_obs)
    begin=time.time()
    # f12_cf = mints.f12_cgtg(gamma)
    QF = np.zeros((nri, nri, no, no))

    # <xy|ij>
    # QF_xyij = mints.ao_f12(f12_cf, bs_cabs, bs_obs, bs_cabs, bs_obs).to_array().swapaxes(1,2)
    # QF_xaij = mints.ao_f12(f12_cf, bs_cabs, bs_obs, bs_obs, bs_obs).to_array().swapaxes(1,2)
    QF_xyij = f12_int.ao_int["f12_cgcg"].swapaxes(1,2)
    QF_xaij = f12_int.ao_int["f12_cggg"].swapaxes(1,2)
    end=time.time()
    print(f"{ sys._getframe(  ).f_code.co_name} time to do integrals in ",end-begin)

    QF[c,c,o,o] = np.einsum("xX,yY,iI,jJ,xyij->XYIJ", Cx, Cx, Cp[:,o], Cp[:,o], QF_xyij, optimize=True)

    # <xa|ij> and <ax|ji>
    
    QF[c,v,o,o] = np.einsum("xX,aA,iI,jJ,xaij->XAIJ", Cx, Cp[:,v], Cp[:,o], Cp[:,o], QF_xaij, optimize=True)
    QF[v,c,o,o] = QF[c,v,o,o].transpose((1,0,3,2))

    G = (0.375 * QF + 0.125 * QF.transpose((0,1,3,2))) / gamma
    print(G.shape)
    return G
@timer_decorator
def gen_V(gamma,sliced_g,my_orbital_space,f12_int):
    bs_obs=my_orbital_space.bs_obs()
    bs_cabs=my_orbital_space.bs_cabs()
    Cp=my_orbital_space.Cp
    Cx=my_orbital_space.Cx
    o=my_orbital_space.o



    being=time.perf_counter()

    rv_gggg=f12_int.ao_int["f12g12_gggg"]
    r_ggga=f12_int.ao_int["f12_gggc"]
    r_gggg=f12_int.ao_int["f12_gggg"] 
    rr_gggg=f12_int.ao_int["f12_squared_gggg"]  
    end=time.perf_counter()
    print(f"{ sys._getframe(  ).f_code.co_name} time to do integrals in ",end-being)
    
    rv_gggg_phy=np.einsum("iajb->ijab",rv_gggg)
    r_ggga_phy=np.einsum("iajb->ijab",r_ggga)
    r_gggg_phy=np.einsum("iajb->ijab",r_gggg)
    rr_gggg_phy=np.einsum("iajb->ijab",rr_gggg)    
    ## generate V term
    # term1 // get mo integral (rv)_{xy}^{ij}
    C_occ=Cp[:,o]
    term1=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rv_gggg_phy,C_occ,C_occ,Cp,Cp,optimize=True)
    # term2 // -r_{xy}^{pq} v_{pq}^{ij} 
    r_xypq=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_gggg_phy,C_occ,C_occ,Cp,Cp,optimize=True)
    v_pqij=sliced_g.mo_int["g_pqrs"]
    term2=np.einsum("xypq,pqij->xyij",r_xypq,v_pqij,optimize=True)
    # term3,term4, -r_{xy}^{a^\prime o} v_{a^prime o ij} -r_{xy}^{ob^\prime}v_{ob^\prime}^{ij}
    r_yxoa=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_ggga_phy,C_occ,C_occ,C_occ,Cx,optimize=True)
    v_jioa=sliced_g.mo_int["g_pqrx"][:,:,o,:]
    term3=np.einsum("yxoa,jioa->yxji",r_yxoa,v_jioa,optimize=True)
    term4=np.einsum("ijkl->jilk",term3)
    V_noper=term1-term2-term3-term4
    ## generate X term together ,to use common imterdiate array
    term1=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rr_gggg_phy,C_occ,C_occ,C_occ,C_occ,optimize=True)
    ## term2  // -r_{xy}^{pq}  the same as those in v term
    term2=np.einsum("xypq,ijpq->xyij",r_xypq,r_xypq,optimize=True)
    ## term3 
    term3=np.einsum("xyob,ijob->xyij",r_yxoa,r_yxoa,optimize=True)
    term4=np.einsum("ijkl->jilk",term3)
    X_noper=term1-term2-term3-term4
    V_noper/=gamma
    X_noper/=gamma**2
    return V_noper,X_noper


@timer_decorator
def gen_b(gamma,my_orbital_space,total_fock,fock_ri_mo,K_ri_mo,f12_int):
    bs_obs=my_orbital_space.bs_obs()
    bs_cabs=my_orbital_space.bs_cabs()
    cp=my_orbital_space.Cp
    Cx=my_orbital_space.Cx
    c=my_orbital_space.c #{x,y,z,...}
    o=my_orbital_space.o #{i,j,k,l,...}
    v=my_orbital_space.v #{a,b,c,d,...}

    cx_save=Cx
    n_occ=o.stop
    n_gbs=my_orbital_space.nbf
    n_cabs=my_orbital_space.ncabs
    n_ri=my_orbital_space.nri




    ## calculation begin here
    begin=time.time()
    d_com_ao=f12_int.ao_int["double_commutator_gggg"]
    d_com_ao_phy=np.einsum("iajb->ijab",d_com_ao)
    d_com_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",d_com_ao_phy,cp[:,o],cp[:,o],
                    cp[:,o],cp[:,o],optimize=True)
    rr_gggc_ao=f12_int.ao_int["f12_squared_gggc"]
    rr_gggg_ao=f12_int.ao_int["f12_squared_gggg"]
    end=time.time()
    print(f"{ sys._getframe(  ).f_code.co_name} time to do integrals in ",end-begin)
    # generat r2 ooo_ri
    rr_gggc_ao_phy=np.einsum("iajb->ijab",rr_gggc_ao)
    rr_gggg_ao_phy=np.einsum("iajb->ijab",rr_gggg_ao)

    rr_ooop_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rr_gggg_ao_phy,
                        cp[:,o],
                        cp[:,o],
                        cp[:,o],
                        cp,optimize=True)
    rr_oooc_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",rr_gggc_ao_phy,
                        cp[:,o],
                        cp[:,o],
                        cp[:,o],
                        cx_save,optimize=True)
    # stack into one term
    rr_ooori=np.concatenate((rr_ooop_mo,rr_oooc_mo),axis=-1)
    temp=np.einsum("mnkP,lP->mnkl",rr_ooori,fock_ri_mo[o,:],optimize=True)
    B_temp=np.copy(d_com_mo)
    B_temp+=temp
    B_temp+=np.einsum("klmn->lknm",temp)
    r_ggga=f12_int.ao_int["f12_gggc"]
    r_gggg=f12_int.ao_int["f12_gggg"]
    r_gaga=f12_int.ao_int["f12_gcgc"]
    r_ggga_phy=np.einsum("iajb->ijab",r_ggga)
    r_gggg_phy=np.einsum("iajb->ijab",r_gggg)
    r_ggaa_phy=np.einsum("iajb->ijab",r_gaga)
    r_oocc_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_ggaa_phy,cp[:,o],cp[:,o],
                        cx_save,cx_save,optimize=True)
    r_oopc_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_ggga_phy,cp[:,o],cp[:,o],
                        cp,cx_save,optimize=True)
    r_oopq_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",r_gggg_phy,cp[:,o],cp[:,o],
                        cp,cp,optimize=True)
    r_oo_ri_ri_mo=np.zeros((n_occ,n_occ,n_ri,n_ri))
    r_oo_ri_ri_mo[:,:,:n_gbs,:n_gbs]=r_oopq_mo
    r_oo_ri_ri_mo[:,:,:n_gbs,n_gbs:]=r_oopc_mo
    r_oo_ri_ri_mo[:,:,n_gbs:,:n_gbs]=np.einsum("ijkl->jilk",r_oopc_mo)
    r_oo_ri_ri_mo[:,:,n_gbs:,n_gbs:]=r_oocc_mo
    temps=np.einsum("mnPQ,PR,klRQ->mnkl",r_oo_ri_ri_mo,K_ri_mo,
            r_oo_ri_ri_mo,optimize=True)
    B_temp-=temps
    B_temp-=np.einsum("mnkl->nmlk",temps)

    def get_CAC_integral(r_oo_ri_ri_mo,total_fock):
        ## fock slice is cc
        ## r slice is CA
        slice_c=slice(0,n_occ)
        slice_a=slice(n_gbs,n_ri)
        return r_oo_ri_ri_mo[:,:,slice_c,slice_a],total_fock[slice_c,slice_c]
    def get_ECE_integral(r_oo_ri_ri_mo,total_fock):
        ## fock slice is EE
        ## r slice is EC
        slice_c=slice(0,n_occ)
        slice_e=slice(0,n_ri)
        return r_oo_ri_ri_mo[:,:,slice_e,slice_c],total_fock[slice_e,slice_e]
    def get_DBD_integral(r_oo_ri_ri_mo,total_fock):
        ## fock slice is DD
        ## r slice is DB
        slice_d=slice(0,n_gbs)
        slice_b=slice(n_occ,n_gbs)
        return r_oo_ri_ri_mo[:,:,slice_d,slice_b],total_fock[slice_d,slice_d]
    B_sym_temp=np.copy(B_temp)
    B_sym_temp-=contraction_of_symmetric(
        *get_ECE_integral(r_oo_ri_ri_mo,total_fock)
    )
    B_sym_temp+=contraction_of_symmetric(
        *get_CAC_integral(r_oo_ri_ri_mo,total_fock)
    )
    B_sym_temp-=contraction_of_symmetric(
        *get_DBD_integral(r_oo_ri_ri_mo,total_fock)
    )
    def get_DBA_integral(r_oo_ri_ri_mo,total_fock):
        slice_d=slice(0,n_gbs)
        slice_b=slice(n_occ,n_gbs)
        slice_a=slice(n_gbs,n_ri)
        return (
            r_oo_ri_ri_mo[:,:,slice_d,slice_b],
            total_fock[slice_d,slice_a],
            r_oo_ri_ri_mo[:,:,slice_a,slice_b]
        )
    def get_EAC_integral(r_oo_ri_ri_mo,total_fock):
        slice_e=slice(0,n_ri)
        slice_a=slice(n_gbs,n_ri)
        slice_c=slice(0,n_occ)
        return (
            r_oo_ri_ri_mo[:,:,slice_e,slice_a],
            total_fock[slice_e,slice_c],
            r_oo_ri_ri_mo[:,:,slice_c,slice_a]
    )
    B_usym_temp=np.copy(B_sym_temp)
    B_usym_temp-=2*contraction_of_unsymmetric(
        *get_EAC_integral(r_oo_ri_ri_mo,total_fock))
    B_usym_temp-=2*contraction_of_unsymmetric(
        *get_DBA_integral(r_oo_ri_ri_mo,total_fock)) ## muliply 2 due to later conj 1/2
    B_final_temp=0.5*B_usym_temp+0.5*np.einsum("mnxy->xymn",B_usym_temp) 
    B_final_temp/=(gamma*gamma)
    return    B_final_temp
def rational_generate(array):
    return 3/8*array+1/8*np.einsum("ijkl->ijlk",array)
def conjugate(array):
    return np.einsum("ijkl->klij",array)
def contraction_of_unsymmetric(r1,f,r2):
    tp=np.einsum("mnpa,pA,xyAa->mnxy",r1,f,r2,optimize=True)
    temp=tp+np.einsum("mnxy->nmyx",tp)
    return temp
def contraction_of_symmetric(r,f):
    tp=np.einsum("mnpa,pq,xyqa->mnxy",r,f,r,optimize=True)
    temp=tp+np.einsum("mnxy->nmyx",tp)
    return temp
