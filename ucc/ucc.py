from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import mqchem
import numpy as np
geometry=[("H",(0,0,0)),("F",(0,0,1.0)) ]
basis="6-31g"
multiplicity=1
charge=0
molecule=MolecularData(geometry,basis,multiplicity,charge)
mol=run_pyscf(molecule,run_ccsd=True,run_fci=False)
print("number of qbit",mol.n_qubits)
print("number of electron",mol.n_electrons)
print("Fci energy",mol.fci_energy)
import numpy as np
h1ct=np.load("h1ct.npy")
h2ct=np.load("h2ct.npy")
import pdb
pdb.set_trace()
hamiltonian,ansatz_circuit,initial_amplitudes=mqchem.prepare_uccsd_vqe(mol,threshold=1e-6)
print("Hamiltonina")
print(str(hamiltonian.fermion_hamiltonian)[:200])

print("ansatz paraname",ansatz_circuit.params_name)
print("number of parameter",len(initial_amplitudes))
print("number of paraname",len(ansatz_circuit.params_name))
#print(initial_amplitudes)

# get ucc
vqe_sim=mqchem.MQChemSimulator(mol.n_qubits,mol.n_electrons)
grad_ops=vqe_sim.get_expectation_with_grad(hamiltonian,ansatz_circuit)
energy,gradient=grad_ops(initial_amplitudes)
print(f"initial energy {energy:8f}")
print(f"initial grad {gradient[:5]}")
print("initial_amplitudes",initial_amplitudes.shape)
from scipy.optimize import minimize
global count
count=0
def print_callback(intermediate_result):
    global count
    count+=1
    fun=intermediate_result.fun
    if count%10==0:
        print(f"{count} energy {fun:.6f}")
from time import time
t0=time()
result=minimize(grad_ops,
        initial_amplitudes,
        method="BFGS",
        jac=True,
        callback=print_callback,
        options={"maxiter":300,"gtol":1e-8,'disp':True})
endtime=time()
print('-'*15+" VQE Information Summary" +'-'*15)
vqe_energy=result.fun
grad=result.jac
vqe_time=round(endtime-t0,3)
print("Optimization Finished")
print('Energy {}\nAmplitudes number :{}\nTime Consuming:{} s\nIteration Number {} \nMax_gnorm{:2f}'.format(
    vqe_energy,
    len(result.x),vqe_time,
            count,np.amax(np.abs(result.jac))))
print(result.message)
#print("final amplitudes",result.x.shape)
#print(result.x)
