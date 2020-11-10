#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG8 
#BSUB -W 00:10 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'

./compile8.sh
mv a.out test8

JSRUN='jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1'

echo $JSRUN
$JSRUN ./test8 -1 > $DATA_PREFIX/8-base.txt
$JSRUN ./test8 0 > $DATA_PREFIX/8-0.txt
$JSRUN ./test8 1 > $DATA_PREFIX/8-1.txt
$JSRUN ./test8 2 > $DATA_PREFIX/8-2.txt
$JSRUN ./test8 3 > $DATA_PREFIX/8-3.txt