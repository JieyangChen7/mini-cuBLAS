
nvcc --ptxas-options=-v -arch=sm_70 dgemm4.cu -lcublas -o test4
