
nvcc --ptxas-options=-v -arch=sm_70 dgemm8.cu -lcublas -o test8
