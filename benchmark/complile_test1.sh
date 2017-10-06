nvcc -O0 -Xcicc -O3 -Xptxas -O3 --ptxas-options=-v -arch=compute_61 -code=compute_61,sm_61 gl_access.cu -lcublas
