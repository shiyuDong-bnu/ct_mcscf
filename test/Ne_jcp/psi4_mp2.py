"""
A reference implementation of second-order Moller-Plesset perturbation theory.

References:
- Algorithms and equations were taken directly from Daniel Crawford's programming website:
http://github.com/CrawfordGroup/ProgrammingProjects

Special thanks to Rob Parrish for initial assistance with libmints.
"""

__authors__    = "Daniel G. A. Smith"
__credits__   = ["Daniel G. A. Smith", "Dominic A. Sirianni", "Rob Parrish"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__   = "BSD-3-Clause"
__date__      = "2017-05-23"

import time
import numpy as np
np.set_printoptions(precision=5, linewidth=200, suppress=True)
import psi4
def compute_mp2(scf_e,eocc,evirt,mo_eri,test=False):
    t = time.time()
    e_denom = 1 / (eocc.reshape(-1, 1, 1, 1) - evirt.reshape(-1, 1, 1) + eocc.reshape(-1, 1) - evirt)
    
    # Get the two spin cases
    MP2corr_OS = np.einsum('iajb,iajb,iajb->',mo_eri , mo_eri, e_denom)
    MP2corr_SS = np.einsum('iajb,iajb,iajb->', mo_eri - mo_eri.swapaxes(1, 3), mo_eri, e_denom)
    print('...MP2 energy computed in %.3f seconds.\n' % (time.time() - t))
    
    MP2corr_E = MP2corr_SS + MP2corr_OS
    MP2_E = scf_e + MP2corr_E
    
    
    print('MP2 SS correlation energy:         %16.10f' % MP2corr_SS)
    print('MP2 OS correlation energy:         %16.10f' % MP2corr_OS)
    
    print('\nMP2 correlation energy:            %16.10f' % MP2corr_E)
    print('MP2 total energy:                  %16.10f' % MP2_E)
    
    if test:
        psi4.set_options({"screening":"csam",
                          "scf_type":"pk",
                          "freeze_core":True})
        psi4.energy('MP2')
        print(psi4.core.variable('MP2 TOTAL ENERGY'), MP2_E)
        print("diff ",MP2_E-psi4.core.variable('MP2 TOTAL ENERGY'))
    return MP2corr_E,MP2_E
if __name__=="__main__":
    # Memory for Psi4 in GB
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    
    # Memory for numpy in GB
    numpy_memory = 2
    
    
    mol = psi4.geometry("""
    Ne 1.0 1.1 1.0
    symmetry c1
    """)
    
    
    psi4.set_options({'basis': 'aug-cc-pvdz',
                      'scf_type': 'pk',
                      'mp2_type': 'conv',
                      "freeze_core":True,
                      'e_convergence': 1e-8,
                      'd_convergence': 1e-8})
    
    # Check energy against psi4?
    check_energy = False
    
    print('\nStarting SCF and integral build...')
    t = time.time()
    
    # First compute SCF energy using Psi4
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)
    
    # Grab data from wavfunction class 
    ndocc = wfn.nalpha()
    nmo = wfn.nmo()
    SCF_E = wfn.energy()
    eps = np.asarray(wfn.epsilon_a())
    
    # Compute size of ERI tensor in GB
    ERI_Size = (nmo ** 4) * 8e-9
    print('Size of the ERI/MO tensor will be %4.2f GB.' % ERI_Size)
    memory_footprint = ERI_Size * 2.5
    if memory_footprint > numpy_memory:
        clean()
        raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                        limit of %4.2f GB." % (memory_footprint, numpy_memory))
    
    print('Building MO integrals.')
    # Integral generation from Psi4's MintsHelper
    t = time.time()
    mints = psi4.core.MintsHelper(wfn.basisset())
    Co = wfn.Ca_subset("AO", "ACTIVE_OCC")
    Cv = wfn.Ca_subset("AO", "VIR")
    MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
    
    Eocc = eps[1:ndocc]
    Evirt = eps[ndocc:]
    
    print('Shape of MO integrals: %s' % str(MO.shape))
    print('\n...finished SCF and integral build in %.3f seconds.\n' % (time.time() - t))
    
    print('Computing MP2 energy...')
    compute_mp2(SCF_E,Eocc,Evirt,MO,test=True)
