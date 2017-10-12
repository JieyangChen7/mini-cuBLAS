git pull github master
nvcc -O0 -Xcicc -O3 -Xptxas -O3 --ptxas-options=-v -arch=sm_35 gl_access.cu -lcublas -o gl_access
#nvcc -O0 -Xcicc -O3 -Xptxas -O3 --ptxas-options=-v -arch=sm_35 fmad.cu -lcublas -o fmad
./gl_access