#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void array_generator(int n, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //clock_t start = clock();
  A[idx] = (unsigned long long int)(A + idx + blockDim.x);
  //clock_t end = clock();
  //printf("%d\n", end-start);
}

__global__ void global_memory(int n, double * A, int space, int iteration, unsigned long long int * T1, unsigned long long int * T2) {
  int idx = blockIdx.x * space + threadIdx.x;
  A = A + idx;
  volatile clock_t start = clock();
  for (int i = 0; i < iteration; i++) {
  	A = (double *)(unsigned long long int) *A;
  }
  volatile clock_t end = clock();
  T1[idx] = start;
  T2[idx] = end;
  //printf("%d ", end-start);
  //printf("SE: %d %d", start, end);

}


int main(){
  int n = 128;
  int B = 16;
  double * A = new double[n + B];
  unsigned long long int * T1 = new unsigned long long int[n];
  unsigned long long int * T2 = new unsigned long long int[n];
  double * dA;
  unsigned long long int *dT1, *dT2, *dT3;
  cudaMalloc(&dA, (n + B) * sizeof(double));
  cudaMalloc((void**)&dT1, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dT2, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dT3, n * sizeof(unsigned long long int));
  array_generator<<<n/B, B>>>(n, dA);
  global_memory<<<n/B, B>>>(n, dA, B, 1, dT1, dT2);
  cudaMemcpy(A, dA, (n + B) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(T1, dT1, n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
  cudaMemcpy(T2, dT2, n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
//  for (int i = 0; i < n + B; i++) {
//  	cout << A[i] << " ";
//  }
    for (int i = 0; i < n; i++) {
    	cout << "" << i << " "<< T1[i] << " " << T2[i] << " " << T2[i] - T1[i] << endl;;
    }
}
