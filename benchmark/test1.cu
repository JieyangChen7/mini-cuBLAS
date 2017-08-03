#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void array_generator(int n, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  clock_t start = clock();
  A[idx] = (unsigned long long int)(A + idx + blockDim.x);
  clock_t end = clock();
  printf("%d\n", end-start);
}


int main(){
  int n = 128;
  int B = 16;
  double * A = new double[n + B];
  double * dA;
  cudaMalloc(&dA, (n + B) * sizeof(double));
  array_generator<<<n/B, B>>>(n, dA);
  cudaMemcpy(A, dA, (n + B) * sizeof(double), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n + B; i++) {
  	cout << A[i] << " ";
  }
}
