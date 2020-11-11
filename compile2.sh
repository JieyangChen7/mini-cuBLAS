
nvcc --ptxas-options=-v -arch=sm_70 dgemm2.cu -lcublas -o test2
