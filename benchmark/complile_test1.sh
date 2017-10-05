nvcc -O0 -Xcicc -O3 -Xptxas -O3 --ptxas-options=-v -arch=compute_35 -code=compute_35,sm_35 gl2r.cu -lcublas
