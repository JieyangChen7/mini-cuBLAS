
nvcc --ptxas-options=-v -arch=sm_70 sgemm16.cu -lcublas -o test16s
