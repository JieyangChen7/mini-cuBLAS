#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG8s 
#BSUB -W 00:10 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'

./compile8s.sh
mv a.out test8s

JSRUN='jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1'

echo $JSRUN
$JSRUN ./test8s -1 > $DATA_PREFIX/8s-base.txt
$JSRUN ./test8s 0 > $DATA_PREFIX/8s-0.txt
$JSRUN ./test8s 1 > $DATA_PREFIX/8s-1.txt
$JSRUN ./test8s 2 > $DATA_PREFIX/8s-2.txt
$JSRUN ./test8s 3 > $DATA_PREFIX/8s-3.txt