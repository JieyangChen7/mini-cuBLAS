#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG4 
#BSUB -W 01:00 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'

./compile4.sh
mv a.out test4

JSRUN='jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1'

echo $JSRUN
$JSRUN ./test4 -1 > $DATA_PREFIX/4-base.txt
$JSRUN ./test4 0 > $DATA_PREFIX/4-0.txt
$JSRUN ./test4 1 > $DATA_PREFIX/4-1.txt
$JSRUN ./test4 2 > $DATA_PREFIX/4-2.txt
$JSRUN ./test4 3 > $DATA_PREFIX/4-3.txt