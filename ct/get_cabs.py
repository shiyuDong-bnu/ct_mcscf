import psi4
from ct.utils.timer import timer_decorator

@timer_decorator
def get_cabs(mol,wfn,basis,df_basis):
    """
    use psi4 to generate cabs basis space 
    """
    # basis = psi4.core.get_global_option('BASIS')
    # df_basis = psi4.core.get_global_option('DF_BASIS_MP2')

    keys = ["BASIS", "CABS_BASIS"]
    targets = [basis, df_basis]
    roles = ["ORBITAL","F12"]
    others = [basis, basis]

    combined = psi4.driver.qcdb.libmintsbasisset.BasisSet.pyconstruct_combined(mol.save_string_xyz(), keys, targets, roles, others)
    combined = psi4.core.BasisSet.construct_from_pydict(mol, combined, combined["puream"])

    obs = wfn.alpha_orbital_space('p', 'SO', 'ALL')
    ribs = psi4.core.OrbitalSpace.build_ri_space(combined, 1.0e-8)
    cabs = psi4.core.OrbitalSpace.build_cabs_space(obs, ribs, 1.0e-6)
    return obs,ribs,cabs
