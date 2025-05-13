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
```
D2[\mu,\lambda,\nu,\kappa]=D^{\mu,\lambda}_{\nu,\kappa}
```
