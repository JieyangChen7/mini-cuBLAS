#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK( fn ) do { \
  CUresult status = (fn); \
  if ( CUDA_SUCCESS != status ) { \
  const char* errstr; \
  cuGetErrorString(status, &errstr); \
  printf("CUDA Driver Failure (line %d of file %s):\n\t%s returned 0x%x (%s)\n", __LINE__, __FILE__, #fn, status, errstr); \
  exit(EXIT_FAILURE); \
  } \
  } while (0)




using namespace std;

void test(int m, int k){
  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  int n = 1;
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

  int threadsPerBlock = 128;
  int blocksPerGrid = m / threadsPerBlock;
  
  void* params[] = {&m, &n, &k, &dA, &lda, &dB, &ldb, &dC, &ldc, &dT};
  
  CUmodule hModule;
  CUDA_CHECK (cuModuleLoad(&hModule, "kernel2.cubin") );
  CUfunction hKernel;
  CUDA_CHECK (cuModuleGetFunction(&hKernel, hModule, "dgemm_kernel2"));

  cuLaunchKernel(hKernel, blocksPerGrid, 1, 1,  threadsPerBlock, 1, 1, 0, 0, params, 0);




  cudaMemcpy(C, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(checkC, dcheckC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(T, dT, m * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

  long long int sum = 0;
  for (int i = 0; i < m; i++) {
    sum += T[i];
    cout << "Thread.id[" << i << "]: latency "<< T[i] << " cycles"<< endl;     
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

int main(int argc, char* argv[])
{
  cuInit(0);

  CUdevice  hDevice;
  CUDA_CHECK( cuDeviceGet(&hDevice, 0) );
  CUcontext hContext = 0;
  CUDA_CHECK( cuCtxCreate(&hContext, 0, hDevice) );

  
  for (int i = 128; i <= 128; i *= 2){
    cout << "Test on: A (" << i << " x " << i << ") by B (" << i << " x " << 1 << ")" << endl;
    test(i, 128);
  }


}



  
