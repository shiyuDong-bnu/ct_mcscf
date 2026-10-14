# CT-UCC

VQE-UCC calculations on a canonical-transformed (CT) Hamiltonian.

## Workflow

1. **Generate the CT Hamiltonian** — run `ct_hf_df.py` first:

   ```bash
   python ct_hf_df.py
   ```

   This performs a Psi4 HF calculation followed by the canonical
   transformation for each H2 bond length, and saves the transformed
   one-/two-electron integrals and MO coefficients as
   `h1ct<R>.npy`, `h2ct<R>.npy`, and `c<R>.npy`.

2. **Run VQE-UCC** — then run `ct_ucc.py`:

   ```bash
   python ct_ucc.py
   ```

   This loads the CT Hamiltonian files generated in step 1, builds the
   qubit Hamiltonian with QC5, and runs VQE-UCC (comparing HF, CCSD,
   UCCSD, and CT-UCCSD energies). Results are saved to `h2_ucc_6_31g.npz`.

Note: `ct_ucc.py` requires the `.npy` files produced by `ct_hf_df.py`,
so the scripts must be run in this order.
