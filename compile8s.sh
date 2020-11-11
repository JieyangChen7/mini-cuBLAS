
nvcc --ptxas-options=-v -arch=sm_70 sgemm8.cu -lcublas -o test8s
