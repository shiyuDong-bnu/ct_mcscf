This Test is used to calculation the properties of ct_mcscf 
dissociation of H2
under the directory we calculate H2 at bond lenght 0.74
```
cd  balence/f12_pt2
```
##To run normal forte CAS(2,2) calculation  following dsrg-pt2
```
python dsrgpt2.py
```
output can be found at bare.out
```
E_{corr}=E_{dsrgpt2}-E_{mcscf}=-0.014367364718224
```
##To run a ct-hf calculation using CAS(2,2) RDM
```
python ct_to_dump.py > dump.out
```
output can be found at  gen_dump.out
ct-hf correlation energy is calculated by
```
E_{f12-hf}-E_{hf}
```
can be found at last line at dump.out file.
The fcidump file in numpy .npy fomrat is generate named by MRDRESSED.DUMP.npy
## To run f12-dsrgpt2  
```
python forte_from_dump.py
```
The correlation energy defined by 
```
E_{f12corr}=E_{f12-dsrgpt2}-E_{mcscf}=-1.168372819268136-(-1.151319160715058)=-0.017053658553078055
```
## the f12 energy is
```
E_{f12corr}-E_{corr}=-0.002686293834854055
```
