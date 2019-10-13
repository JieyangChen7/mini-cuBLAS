#include <stdlib.h>
#include <iostream>
#include "cublas_v2.h"
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <cuda_fp16.h>
#define TEST_RUN 10 
#define ESP 10e-10
using namespace std;


void check_cuda_error(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
}

void check_C(float * dC, int m, int n, float * checkC) {
  for (int i = 0; i < m * n; i++){
    //cout << i << endl;
    if (fabs(dC[i] - checkC[i]) > ESP){
      cout << "error:" << fabs(dC[i] - checkC[i]) << endl;
      return;
    }
  }
  cout << "correct" << endl;
}


float test_cublas_mm(int m, int n, int k, 
            __half * dA, int lda, 
            __half * dB, int ldb, 
            __half * dC, int ldc);


void test(int m, int k);

int main(){
  for (int i = 10240; i <= 30720; i += 1024){
  //int i = 1024;
    cout << "Test on: A (" << i << " x " << i << ") by B (" << i << " x " << 2 << ")" << endl;
    test(i, i);
  }
}

void test(int m, int k){
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    //int m = 20480;
    int n = 2;
    //int k = 20480;
    __half * A = new __half[m * k];
    __half * B = new __half[n * k];
    __half * C = new __half[m * n];
    __half * checkC = new __half[m * n];     

    for (int i = 0; i < m * k; i++){
    	A[i] = __float2half((float)i/(m * k));
    }

    for (int i = 0; i < n * k; i++){
    	B[i] = __float2half((float)i/(n * k));
    }
    
    __half * dA;
    cudaMalloc(&dA, m * k * sizeof(__half));
    int lda = m;

    __half * dB; 
    cudaMalloc(&dB,  n * k * sizeof(__half));
    int ldb = k;

    __half * dC;
    cudaMalloc(&dC, m * n * sizeof(__half));
    int ldc = m;

    __half * dcheckC;
    cudaMalloc(&dcheckC, m * n * sizeof(__half));

    cudaMemcpy(dA, A, m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, n * k * sizeof(__half), cudaMemcpyHostToDevice);
    
    float base;

    base = test_cublas_mm(m, n, k,  dA, lda, dB, ldb, dcheckC, ldc);
  
   
   
    cudaMemcpy(C, dC ,m * n * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaMemcpy(checkC, dcheckC, m * n * sizeof(__half), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < m * n; i++){
    // cout<<C[i]<<" ";	
    //}
    //check_C(C, m, n, checkC);

    //free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] checkC;

}



float test_cublas_mm(int m, int n, int k, 
         	     __half * dA, int lda, 
                     __half * dB, int ldb, 
                     __half * dC, int ldc){

    __half one = __float2half(1.0);
    __half zero = __float2half(0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TEST_RUN; i++)
      cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        &one, dA, lda, dB, ldb, &zero, dC, ldc);
      check_cuda_error();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float real_time = milliseconds / 1000;

    cout <<"Runing time of culashgemm:" << real_time <<" s." << endl;
    return real_time;
}






























