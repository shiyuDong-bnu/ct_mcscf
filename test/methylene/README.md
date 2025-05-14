This dicectory contains scripts  to do  calculation of CH2.

**dsrtpt2_sig.py** and **dsrgpt2_trip.py** is used to extrapolate basis set limit 

## CT-DSRTPT2 is at directory test_dump/basis_scan

```bash
cd test_dump/basis_scan
cd dz
```
dz is calculation using cc-pvdz-f12 basis set 
## Calculate mcscf and CT-HF to dump integral
```
python ct_to_dump.py
```
This will generate **gen_dump.out** and  **MRDRESSED.DUMP.npy**
## 
## Calculate CT-MCSCF-DSRG-PT2 run
```bash
python forte_from_dump.py
```
This will generate **dresse.out**
## Calculate MCSCF-DSRG-PT2 run
```
python dsrgpt2.py
```

## other directory
tz ,qz use  cc-pvtz-f12 and cc-pvqz-f12 respectivly
trip_dz ,trip_tz trip_qz  calculate CH2 triplet state 

