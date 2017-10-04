nvcc -O0 -Xcicc -O3 -Xptxas -O2 --ptxas-options=-v -arch=compute_35 -code=compute_35,sm_35 kernel2.cu -lcublas
