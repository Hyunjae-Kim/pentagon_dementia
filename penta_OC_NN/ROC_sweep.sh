#!/bin/bash

for (( idx1=0; idx1<4; idx1++)) 
do
for (( idx0=0; idx0<10; idx0++)) 
do


	python OC_NN_ROC_sweep.py ${idx0} ${idx1}
    
done
done
