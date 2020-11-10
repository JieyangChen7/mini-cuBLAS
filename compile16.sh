
nvcc --ptxas-options=-v -arch=sm_70 dgemm16.cu -lcublas
