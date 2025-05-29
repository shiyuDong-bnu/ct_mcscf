## To Do List
> [!IMPORTANT]
1.  The integral cannot store in computing benzene.
2. CABS singlet is not included
3. change the workflow , do not do ct-mcscf anymore.
## Usage
First 
Install psi4 and ambit following https://forte.readthedocs.io/en/latest/nb_01_compilation.html

Download forte from [git@github.com:shiyuDong-bnu/forte.git](https://github.com/shiyuDong-bnu/forte.git)
```
git clone git@github.com:shiyuDong-bnu/forte.git
git checkout ct_forte
```
Next 
Download ct_mcscf from [git@github.com:shiyuDong-bnu/ct_mcscf.git](https://github.com/shiyuDong-bnu/ct_mcscf.git)
```
git clone git@github.com:shiyuDong-bnu/ct_mcscf.git
```
Examples are in test 
```
cd test
```

Information of 2RDM order in ct
eq(#2)
```math
\hat E_{\nu \kappa}^{\mu\lambda}=\sum_{\sigma \tau=\alpha,\beta} \hat a_{\mu\sigma}^\dagger \hat a _{\lambda \tau}^\dagger 
\hat a _{\kappa \tau}\hat a_{\nu\sigma} 
```
eq(#4)


```math
D^{\mu\lambda}_{\nu\kappa}=\langle \Phi_0 \vert \hat E_{\nu \kappa}^{\mu\lambda} \vert \Phi_0 \rangle
```
In implementation we use numpy array  ,the index is
```math
D2[\mu,\lambda,\nu,\kappa]=D^{\mu,\lambda}_{\nu,\kappa}
```
In forte
 The spin-dependent n-body RDMs are defined by
 *
 *   D(p1,σ1; ...; pn,σn; q1,σ1; ...; qn,σn) = <A| a+(p1,σ1) ... a+(pn,σn) a(qn,σn) ... a(q1,σ1) |B>
 *
 * where spins σ1 >= σ2 >= ... >= σn given that α > β.
 * The spin-free n-body RDMs are defined using spin-dependent RDMs as
 *
 *   F(p1, ..., pn, q1, ..., qn) = sum_{σ1,...,σn} D(p1,σ1; ...; pn,σn; q1,σ1; ...; qn,σn)
 *
