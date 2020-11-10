#!/bin/bash 
### Begin BSUB Options 
#BSUB -P csc143 
#BSUB -J MG1 
#BSUB -W 01:00 
#BSUB -nnodes 1 
#BSUB -alloc_flags "smt1"

DATA_PREFIX='/gpfs/alpine/scratch/jieyang/csc143/tsm2'



./compile$1.sh
echo $JSRUN
jsrun -n1 -a 1 -c 1 -g 6 -r 1 -l CPU-CPU -d packed -b packed:1 --smpiargs="-disable_gpu_hooks" ./a.out > $DATA_PREFIX/$1.txt




rm -rf $DATA_PREFIX/$NPROC
mkdir -p $DATA_PREFIX/$NPROC
mkdir -p $DATA_PREFIX/$NPROC/cpu
mkdir -p $DATA_PREFIX/$NPROC/cuda

#$JSRUN js_task_info | sort
$JSRUN ./build/test $N 10 $DATA_PREFIX/$NPROC/cpu/ 0 
$JSRUN ./build/test $N 10 $DATA_PREFIX/$NPROC/cuda/ 1