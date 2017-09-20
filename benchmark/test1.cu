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

__global__ void global_memory(int n, double * A, int space, int iteration, unsigned long long int * T) {
  int idx = blockIdx.x * space + threadIdx.x;
  A = A + idx;
  volatile clock_t start = 0;
  volatile clock_t end = 0;
  volatile unsigned long long sum_time = 0;

  for (int i = 0; i < iteration; i++) {
    start = clock();
  	A = (double *)(unsigned long long int) *A;
    end = clock();
    sum_time += (end - start);
  }
  T[idx] = sum_time;

  //printf("%d ", end-start);
  //printf("SE: %d %d", start, end);

}

__global__ void tid_time(int iteration, unsigned long long int * T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  volatile clock_t start = 0;
  volatile clock_t end = 0;
  unsigned long long sum_time = 0;
  int idx2 = 0;
  int a = blockIdx.x;
  int b = blockDim.x;
  int c = threadIdx.x;
  for (int i = 0; i < iteration; i++) {
    start = clock();
    idx2 += a * b + c;
    end = clock();
    sum_time += (end - start);
  }
  T[idx] = sum_time;

  //printf("%d ", end-start);
  //printf("SE: %d %d", start, end);

}



int main(){
  int n = 128;
  int B = 16;
  double * A = new double[n + B];
  unsigned long long int * T = new unsigned long long int[n];
  
  double * dA;
  unsigned long long int *dT;
  cudaMalloc(&dA, (n + B) * sizeof(double));
  cudaMalloc((void**)&dT, n * sizeof(unsigned long long int));

  //array_generator<<<n/B, B>>>(n, dA);
  //global_memory<<<n/B, B>>>(n, dA, B, 1, dT);
  tid_time<<<n/B, B>>>(1, dT);
  cudaMemcpy(A, dA, (n + B) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(T, dT, n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
 
//  for (int i = 0; i < n + B; i++) {
//  	cout << A[i] << " ";
//  }
    for (int i = 0; i < n; i++) {
    	cout << "" << i << " "<< T[i] << endl;;
    }
}
