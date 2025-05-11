import numpy as np
def write_dump(fname,n_ele,n_obs,V_nuc,mo_h_core,mo_eri):
    with open(fname,"w") as dump_file:
        header="""&FCI
NORB={n_obs},
NELEC={n_ele},
MS2=0,
UHF=.FALSE.,
ORBSYM={orbsym}
ISYM=1,
&END""".format(n_obs=n_obs,n_ele=n_ele,orbsym="1,"*n_obs)
        print(header,file=dump_file)
        n_comp=(n_obs+1)*n_obs//2
        tri_ind=np.tril_indices(n_obs)
        for i in range(n_comp):
            for j in range(0,i+1):
                ind1=tri_ind[0][i]
                ind2=tri_ind[1][i]
                ind3=tri_ind[0][j]
                ind4=tri_ind[1][j]
                val=mo_eri[ind1,ind2,ind3,ind4]
                if abs(val)>1e-10:
                    print("{:28.20E}{:4d}{:4d}{:4d}{:4d}".format(val,ind1+1,ind2+1,ind3+1,ind4+1),file=dump_file)
        for i in range(n_obs):
            for j in range(i+1):
                    val=mo_h_core[i,j]
                    if abs(val)>1e-10:
                        print("{:28.20E}{:4d}{:4d}{:4d}{:4d}".format(val,i+1,j+1,0,0),file=dump_file)
        print("{:28.20E}{:4d}{:4d}{:4d}{:4d}".format(V_nuc,0,0,0,0),file=dump_file)
def write_dump_np(fname,n_ele,ms2,n_obs,V_nuc,mo_h_core,mo_eri):
    myfcidump={}
    myfcidump['norb']=n_obs
    myfcidump['nelec']=n_ele
    myfcidump['ms2']=ms2
    myfcidump['uhf']=False
    myfcidump['orbsym']=[1]*n_obs
    myfcidump['isym']=1
    myfcidump['enuc']=V_nuc
    myfcidump['hcore']=mo_h_core
    myfcidump['eri']=mo_eri
    np.save(fname,myfcidump)
