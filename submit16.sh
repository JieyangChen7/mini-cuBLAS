#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG16 
#BSUB -W 01:00 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'

./compile16.sh
mv a.out test16

JSRUN='jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1'

echo $JSRUN
$JSRUN ./test16 -1 > $DATA_PREFIX/16-base.txt
$JSRUN ./test16 0 > $DATA_PREFIX/16-0.txt
$JSRUN ./test16 1 > $DATA_PREFIX/16-1.txt
$JSRUN ./test16 2 > $DATA_PREFIX/16-2.txt
$JSRUN ./test16 3 > $DATA_PREFIX/16-3.txt