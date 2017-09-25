nvcc --ptxas-options=-v -arch=compute_35 -code=compute_35,sm_35 test1.cu -lcublas
