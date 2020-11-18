#!/bin/bash
VOLTA='-gencode arch=compute_70,code=sm_70 '
TURING='-gencode arch=compute_75,code=sm_75 '
AMP='-gencode arch=compute_80,code=sm_80 '
NVCC_FLAG=$VOLTA$TURING$AMP

nvcc --ptxas-options=-v $NVCC_FLAG dgemm2.cu -lcublas -o test2
nvcc --ptxas-options=-v $NVCC_FLAG dgemm4.cu -lcublas -o test4
nvcc --ptxas-options=-v $NVCC_FLAG dgemm8.cu -lcublas -o test8
nvcc --ptxas-options=-v $NVCC_FLAG dgemm16.cu -lcublas -o test16

nvcc --ptxas-options=-v $NVCC_FLAG sgemm2.cu -lcublas -o test2s
nvcc --ptxas-options=-v $NVCC_FLAG sgemm4.cu -lcublas -o test4s
nvcc --ptxas-options=-v $NVCC_FLAG sgemm8.cu -lcublas -o test8s
nvcc --ptxas-options=-v $NVCC_FLAG sgemm16.cu -lcublas -o test16s