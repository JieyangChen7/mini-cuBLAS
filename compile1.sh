
#nvcc --ptxas-options=-v -arch=sm_35 dgemm1.cu -lcublas

nvcc --ptxas-options=-v -arch=sm_35 -Xptxas -O0 dgemm1.cu -lcublas
