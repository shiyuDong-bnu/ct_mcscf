This directory is used to 
check the corretness of fcidump

We first run an mcscf ,and save the integral in fcidump
then we read the integral ,if the optimization is done in zero step
we say that fcidump is correct.
```
python forte_mcscf.py 
``` 
This will run a mcscf in forte ,and dump the hamiltonian to TEST.DUMP.npy
Return energy(): -38.94082208693549

Next we run 
```
python forte_dump.py
```
This time the forte read integral from TEST.DUMP.npy
Return energy(): -38.94082208693524

Under the trip_dump we test triplet state and the usage is similar
