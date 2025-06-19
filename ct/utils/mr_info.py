from dataclasses import dataclass
import numpy as np


@dataclass
class MRInfo:
    n_frozen: int
    n_restricted_docc: int
    n_active: int
    n_virtual: int
    rdm1: np.ndarray  # 1rdm
    rdm2: np.ndarray

    def __init__(self, orbinf, rdm1, rdm2):
        self.n_frozen = orbinf["n_frozen"]
        self.n_restricted_docc = orbinf["n_restricted_docc"]
        self.n_active = orbinf["n_active"]
        self.n_virtual = orbinf["n_virtual"]
        self.rdm1 = rdm1
        self.rdm2 = rdm2
        self.active_index = slice(
            self.n_frozen + self.n_restricted_docc,
            self.n_frozen + self.n_restricted_docc + self.n_active,
        )
        self.inactive_docc_index = slice(0, self.n_frozen + self.n_restricted_docc)
        self.occupied_index = slice(
            0, self.n_frozen + self.n_restricted_docc + self.n_active
        )

    @property
    def n_occupied(self):
        """
        total number of occupied orbitals
        """
        return self.n_frozen + self.n_restricted_docc + self.n_active

    @property
    def nbf(self):
        """
        total number of basis functions
        """
        return self.n_occupied + self.n_virtual

    @property
    def o(self):
        """
        occ C {i,j,k,l,...}
        """
        return slice(0, self.n_occupied)

    @property
    def v(self):
        """
        vir in gbs B {a,b,c,d,...}
        """
        return slice(self.n_occupied, self.nbf)

    def __repr__(self):
        return (
            f"MRInfo(n_frozen={self.n_frozen}, "
            f"n_restricted_docc={self.n_restricted_docc}, "
            f"n_active={self.n_active}, "
            f"n_virtual={self.n_virtual}"
            f"n_occupied={self.n_occupied}, "
            f"n_basis_functions={self.nbf}, )"
        )
