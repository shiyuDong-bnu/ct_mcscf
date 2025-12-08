import numpy  as np
class RDM():
    def __init__(self,my_orbital_space,mr_info=None):
        self.my_orbital_space=my_orbital_space
        self.nbf=my_orbital_space.nbf
        self.mr_info=mr_info
        self.rdm1=None
        self.rdm2=None
        self.gen_rdm()
    def gen_rdm(self):
        delta = np.identity(self.nbf)
        D1 = np.zeros((self.nbf, self.nbf))
        no=self.my_orbital_space.o.stop
        o=self.my_orbital_space.o
        D2 = np.zeros((no,no,no,self.nbf))
        if self.mr_info is None:
            ## single reference closed shell
            o=self.my_orbital_space.o   
            D1[o,o] = 2 * delta[o,o]

            D2[o,o,o,o] = 4 * np.einsum("ij,kl->ikjl", delta[o,o], delta[o,o])
            D2[o,o,o,o] -= 2 * np.einsum("il,kj->ikjl", delta[o,o], delta[o,o])
        if self.mr_info is  not None:
            ## multireference
            a_ind=self.mr_info.active_index
            o_ind=self.mr_info.inactive_docc_index
            ## 1rdm
            D1[o_ind,o_ind] = 2 * delta[o_ind,o_ind]
            D1[a_ind,a_ind]=self.mr_info.rdm1
            ## end copy from single reference

            ## 2rdm ,first index is occ
            D2[o_ind,:,:,:]=2*np.einsum("iq,rs->irqs",delta[o_ind,:],D1)[o,o,o,:]
            D2[o_ind,:,:,:]-=np.einsum("is,rq->irqs",delta[o_ind,:],D1)[o,o,o,:]
            ## first index is active
            ## second index is occ
            D2[a_ind,o_ind,:,:]=2*np.einsum("is,uq->uiqs",delta[o_ind,:],D1[a_ind,:])[o,o,o,:]
            D2[a_ind,o_ind,:,:]-=np.einsum("iq,us->uiqs",delta[o_ind,:],D1[a_ind,:])[o,o,o,:]
            ## second index is active
            D2[a_ind,a_ind,a_ind,a_ind]=self.mr_info.rdm2
        self.rdm1=D1
        self.rdm2=D2
    def form_fock_density(self):
        o=self.my_orbital_space.o   
        return self.rdm1[o,o]
    def form_d_bar_oooo(self):
        # Eq. (28)
        D1=self.rdm1
        D2=self.rdm2
        o=self.my_orbital_space.o
        Dbar = 2 * np.einsum("pq,rs->prqs", D1[o,o], D1[o,o]) - np.einsum("ps,rq->prqs", D1[o,o], D1[o,o]) - D2[o,o,o,o]
        return Dbar

    def form_d_bar_ooos(self):
        # Eq. (28)
        D1=self.rdm1
        D2=self.rdm2
        o=self.my_orbital_space.o
        s=self.my_orbital_space.s
        Dbar = 2 * np.einsum("pq,rs->prqs", D1[o,o], D1[o,s]) - np.einsum("ps,rq->prqs", D1[o,s], D1[o,o]) - D2[o,o,o,s]
        return Dbar

