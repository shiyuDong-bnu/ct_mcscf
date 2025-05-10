This directory compare the  F12-HF and F12-MP2  result
 with J. Chem. Phys. 158, 057101 (2023)
The difference is atrributed to gassuain fitting  of f12 geminal 
```math
e^{-\lambda r}\approx \sum_i c_i e^{-\lambda_i r^2}
```

To run the scripts ,you need to get correct program path
```python
import sys
sys.path.append("/data/home/sydong/work/ct_mcscf/")  ## change the path to your ct_mcscf path
sys.path.append("/data/home/sydong/software/forte/")  ## change the path to your ct_forte path
```
For F12-HF
 just run 
```
python ct_hf.py  > ref.out 
grep correlation ref.out
```
For F12-MP2
```
python  ct_mp2.py > ref.out
grep corr ref.out
```
