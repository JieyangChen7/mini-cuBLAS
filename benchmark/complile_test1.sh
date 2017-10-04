nvcc -O0 -Xcicc -O0 -Xptxas -O1 --ptxas-options=-v -arch=compute_35 -code=compute_35,sm_35 test1.cu -lcublas
