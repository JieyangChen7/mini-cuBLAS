#include <stdlib.h>
#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void
dgemm_kernel2(int m, int n, int k, double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  //determine the row to process                          
  clock_t start = clock();
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  clock_t end = clock() - start;
  end += idx;
  end -= idx;

  A = A + idx;
  double a;
  double b;
  double temp = 0;
  for (int i = 0;i < k; i++){
    //a = *(A + i * lda);
    //b = *(B + i);
    a = *A;
    b = *B;
    A = A + lda;
    B = B + 1;
    temp = temp + a * b;
  }
  *(C + idx) = temp;
  printf("%d\n", end);
}


void test_kernel2(int m, int n, int k, 
		  double * dA, int lda, 
		  double * dB, int ldb, 
		  double * dC, int ldc){


  int T = 128;
  int blocksPerGrid = m / T;
  int threadsPerBlock = T;
    
  clock_t t = clock();

  for (int i = 0; i < 1; i++)
    dgemm_kernel2<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, 
						      dA, lda, dB, ldb, dC, ldc);


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

  for (int i = 0;i < m * k; i++){
    A[i] = i;
  }

  //    for (int i = 0; i < m; i++){
  // for (int j = 0; j < k; j++){
  //cout << *( A + i + j * m) << " ";
  // }
  // cout << endl;
  //}
    
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

  cudaMemcpy(dA, A, m * k * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, n * k * sizeof(double), cudaMemcpyHostToDevice);
    
  
 
  test_kernel2(m, n, k, dA, lda, dB, ldb, dC, ldc);
 
   
  cudaMemcpy(C, dC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(checkC, dcheckC, m * n * sizeof(double), cudaMemcpyDeviceToHost);

  //check_C(C, m, checkC);    
    
  //for (int i = 0; i < m * n; i++){
  // cout<<C[i]<<" ";
  //}
    
  //free device memory
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] checkC;

}

int main(){
  for (int i = 128; i <= 128; i *= 2){
    cout << "Test on: A (" << i << " x " << i << ") by B (" << i << " x " << 1 << ")" << endl;
    test(i, i);
  }
}
