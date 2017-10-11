nvcc -O0 -Xcicc -O3 -Xptxas -O3 --ptxas-options=-v -arch=sm_35 gl_access.cu -lcublas
