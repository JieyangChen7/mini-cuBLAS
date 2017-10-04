#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
using namespace std;
extern "C"
__global__ void
dgemm_kernel2(int m, int n, int k, double * A, int lda, double * B, int ldb, double * C, int ldc, unsigned long long int * T)
{
  register clock_t start;
  register clock_t end;
  //determine the row to process                          
  //start = clock();
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  A = A + idx;
  register double a;
  register double b;
  register double temp = 0;
  start = clock();
  #pragma unroll 1
  for (int i = 0;i < k; i++){
    a = *A;
    b = *B;
    A = A + lda;
    B = B + 1;
    temp = temp + a * b;
  }
  end = clock() - start;
  *(C + idx) = temp;
  //end = clock() - start;
  T[idx] = end;
}
/*
__global__ void
dgemm_kernel2_sass(int m, int n, int k, double * A, int lda, double * B, int ldb, double * C, int ldc, unsigned long long int * T)
{
  register clock_t start;
  register clock_t end;
  //determine the row to process
  start = clock();
  
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  register double a;
  register double b;
  register double * AA =A;
  register double * BB = B;
  register double temp = 0;
  register long long int lda1 = lda * 8;
  
  #pragma unroll 1
  for (int i = 0;i < k; i++){
    //start = clock();
    asm volatile ("{\n\t"
                  "ld.global.f64 %0, [%2];\n\t"
                  "ld.global.f64 %1, [%3];\n\t"
                  "add.u64 %2, %2, %5;\n\t"
		  "add.u64 %3, %3, 0x8;\n\t"
		  "fma.rz.f64 %4,%0,%1,%4;\n\t"
                  "}"
                  : "+d"(a), "+d"(b), "+l"(AA), "+l"(BB),
		    "+d"(temp):"l"(lda1): "memory"
		  );
    //end = clock() - start; 
  }
  *(C + idx) = temp;
  end = clock() - start;
  T[idx] = end;
}
*/

void test_kernel2(int m, int n, int k, 
		  double * dA, int lda, 
		  double * dB, int ldb, 
		  double * dC, int ldc,
		  unsigned long long int * dT){


  int T = 128;
  int blocksPerGrid = m / T;
  int threadsPerBlock = T;
    
  clock_t t = clock();

  for (int i = 0; i < 1; i++){
    dgemm_kernel2<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, dA, lda, dB, ldb, dC, ldc, dT);
    //dgemm_kernel2_sass<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, dA, lda, dB, ldb, dC, ldc, dT); 
  }


  cudaDeviceSynchronize();
  t = clock() - t;
  float real_time = ((float)t)/CLOCKS_PER_SEC;

  cout <<"Runing time of dgemm_kernel2: " << real_time << " ms." << endl;    

} 


void test(int m, int k){
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  //int m = 20480;
  int n = 1;
  //int k = 20480;
  double * A = new double[m * k];
  double * B = new double[n * k];
  double * C = new double[m * n];  
  double * checkC = new double[m * n];   
  unsigned long long int * T = new unsigned long long int[m];
  
  for (int i = 0;i < m * k; i++){
    A[i] = i;
  }    
  for (int i = 0; i < n * k; i++){
    B[i] = 1;
  }


    
  double * dA;
  cudaMalloc(&dA, m * k * sizeof(double));
  int lda = m;

  double * dB; 
  cudaMalloc(&dB,  n * k * sizeof(double));
  int ldb = k;

  double * dC;
  cudaMalloc(&dC, m * n * sizeof(double));
  int ldc = m;

  double * dcheckC;
  cudaMalloc(&dcheckC, m * n * sizeof(double));

  unsigned long long int *dT;
  cudaMalloc((void**)&dT, m * sizeof(unsigned long long int));

  cudaMemcpy(dA, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    
  
 
  test_kernel2(m, n, k, dA, lda, dB, ldb, dC, ldc, dT);
 
   
  cudaMemcpy(C, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(checkC, dcheckC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(T, dT, m * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  
  long long int sum = 0;
  for (int i = 0; i < m; i++) {
    sum += T[i];
    //cout << "Thread.id[" << i << "]: latency "<< T[i] << " cycles"<< endl;
  }
  cout << "Average latency is: " << (double)sum / m << "cycles" << endl;


  //free device memory
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dT);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] checkC;
  delete[] T;

}

int main(){
  for (int i = 128; i <= 128; i *= 2){
    cout << "Test on: A (" << i << " x " << i << ") by B (" << i << " x " << 1 << ")" << endl;
    test(i, 128);
  }
}
