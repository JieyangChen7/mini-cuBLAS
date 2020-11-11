#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG2
#BSUB -W 01:00 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'

./compile2.sh

JSRUN='jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1'

echo $JSRUN
$JSRUN ./test2 -1 > $DATA_PREFIX/2-base.txt
$JSRUN ./test2 0 > $DATA_PREFIX/2-0.txt
$JSRUN ./test2 1 > $DATA_PREFIX/2-1.txt
$JSRUN ./test2 2 > $DATA_PREFIX/2-2.txt
$JSRUN ./test2 3 > $DATA_PREFIX/2-3.txt