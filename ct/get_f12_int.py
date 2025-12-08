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
from numba import njit, prange,set_num_threads
# G = 3/8 <αβ|Q12 F12|ij> + 1/8 <αβ|Q12 F12|ji>
@timer_decorator
def get_f12(my_orbital_space,f12_int,gamma):
    nri=my_orbital_space.nri
    c=my_orbital_space.c
    o=my_orbital_space.o
    v=my_orbital_space.v
    no=o.stop
    ## calculation begin here
    QF = np.zeros((nri, nri, no, no))
    f12_int.form_f12_moint()    
    QF[c,c,o,o]=f12_int.mo_int["r_xyij"]
    QF[c,v,o,o]=f12_int.mo_int["r_xaij"]
    QF[v,c,o,o]=f12_int.mo_int["r_xaij"].transpose((1,0,3,2))
    G = (0.375 * QF + 0.125 * QF.transpose((0,1,3,2))) / gamma
    print(G.shape)
    return G
@timer_decorator
def gen_V(gamma,sliced_g,my_orbital_space,f12_int):

    o=my_orbital_space.o
    ## load mo integral
   
    f12_int.form_v_and_x_moint()
    rv_ijpq=f12_int.mo_int["rv_ijpq"]
    r_ijpq=f12_int.mo_int["r_ijpq"]
    r_ijoa=f12_int.mo_int["r_ijoa"]
    rr_ijkl=f12_int.mo_int["rr_ijkl"]
    #v_pqij=sliced_g.mo_int["g_pqrs"]
    v_jioa=np.moveaxis(sliced_g.mo_int["g_pixq"],[0,1,2,3],[1,2,3,0])[:,:,o,:]

    L1=sliced_g.mo_int["C_mo_pq"]
    R1=sliced_g.mo_int["T_ind_pq"]
    L1_shaped=L1.reshape(L1.shape[0]*L1.shape[1],L1.shape[2])
    R1_shaped=R1.reshape(R1.shape[0]*R1.shape[1],R1.shape[2])
    
   
   # term1 // get mo integral (rv)_{xy}^{ij}
    term1=rv_ijpq
    # term2 // -r_{xy}^{pq} v_{pq}^{ij} 
    #term2=np.einsum("xypq,pqij->xyij",r_ijpq,v_pqij,optimize=True)
    # term2 df
    # r_{ij}^{pq} g_{pq}_{rs} =r_{ij}^{pq} L^A_{pr} R^{A}_{qs}
    @njit(parallel=True)
    def df_term2():
        term2_df=np.zeros_like(term1)
        for i in range(term1.shape[0]):
            for j in range(term1.shape[1]):
                r_ijpq_slice=np.ascontiguousarray(r_ijpq[i,j,:,:])
                #temp1=np.einsum("pq,Apr->Aqr",r_ijpq_slice,L1)
                #temp1=np.einsum("pq,Arp->Aqr",r_ijpq_slice,L1)
                #print(r_ijpq_slice.flags['C_CONTIGUOUS'])
                #print(L1_shaped.flags['C_CONTIGUOUS'])
                temp1_dot=np.dot(r_ijpq_slice.T,L1_shaped.T)#  q,p || p,Ar -> q,Ar
                temp1_dot=temp1_dot.reshape(r_ijpq_slice.shape[0],L1.shape[0],L1.shape[1]) # q,A,r
                temp1_dot=np.ascontiguousarray(np.swapaxes(temp1_dot,1,0))
                #temp1_dot=np.swapaxes(temp1_dot,1,0)  # A,q,r
                #print(np.allclose(temp1_dot,temp1),end=" ")
                temp1_dot=temp1_dot.reshape(temp1_dot.shape[0]*temp1_dot.shape[1],temp1_dot.shape[2]) # Aq,r
                #print(np.allclose(temp1_dot.reshape(*L1.shape),temp1),end=" ")
                #if not np.allclose(temp1_dot.reshape(*L1.shape),temp1):
                #    import pdb
                #    pdb.set_trace()
                #    pass

                temp2_dot=np.dot(temp1_dot.T,R1_shaped)
                #temp2=np.einsum("Aqr,Aqs->rs",temp1,R1)
                #print(np.allclose(temp2_dot,temp2))
                term2_df[i,j,:,:]=temp2_dot
                #term2_df[i,j,:,:]=temp2
        return term2_df
    set_num_threads(64)
    term2=df_term2()
    # term3,term4, -r_{xy}^{a^\prime o} v_{a^prime o ij} -r_{xy}^{ob^\prime}v_{ob^\prime}^{ij}

    term3=np.einsum("yxoa,jioa->yxji",r_ijoa,v_jioa,optimize=True)
    term4=np.einsum("ijkl->jilk",term3)
    V_noper=term1-term2-term3-term4
    ## generate X term together ,to use common imterdiate array
    term1=rr_ijkl
    ## term2  // -r_{xy}^{pq}  the same as those in v term
    term2=np.einsum("xypq,ijpq->xyij",r_ijpq,r_ijpq,optimize=True)
    ## term3 
    term3=np.einsum("xyob,ijob->xyij",r_ijoa,r_ijoa,optimize=True)
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
    f12_int.form_b_moint()

    rr_ooori=f12_int.mo_int["rr_ooori"]
    d_com_mo=f12_int.mo_int["d_com_mo"]
    temp=np.einsum("mnkP,lP->mnkl",rr_ooori,fock_ri_mo[o,:],optimize=True)
    B_temp=np.copy(d_com_mo)
    B_temp+=temp
    B_temp+=np.einsum("klmn->lknm",temp)

    r_oo_ri_ri_mo=f12_int.mo_int["r_oo_ri_ri_mo"]
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
