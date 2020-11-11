#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG2s 
#BSUB -W 01:00 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'

./compile2s.sh

JSRUN='jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1'

echo $JSRUN
$JSRUN ./test2s -1 > $DATA_PREFIX/2s-base.txt
$JSRUN ./test2s 0 > $DATA_PREFIX/2s-0.txt
$JSRUN ./test2s 1 > $DATA_PREFIX/2s-1.txt
$JSRUN ./test2s 2 > $DATA_PREFIX/2s-2.txt
$JSRUN ./test2s 3 > $DATA_PREFIX/2s-3.txt