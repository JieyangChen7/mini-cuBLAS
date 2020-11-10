#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG16s
#BSUB -W 01:00 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'

./compile16s.sh
mv a.out test16s

JSRUN='jsrun -n 1 -a 1 -c 1 -g 1 -r 1 -l CPU-CPU -d packed -b packed:1'

echo $JSRUN
$JSRUN ./test16s -1 > $DATA_PREFIX/16s-base.txt
$JSRUN ./test16s 0 > $DATA_PREFIX/16s-0.txt
$JSRUN ./test16s 1 > $DATA_PREFIX/16s-1.txt
$JSRUN ./test16s 2 > $DATA_PREFIX/16s-2.txt
$JSRUN ./test16s 3 > $DATA_PREFIX/16s-3.txt