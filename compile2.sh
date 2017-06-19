
nvcc --ptxas-options=-v –X ptxas –dlcm=cg dgemm2.cu -lcublas 
