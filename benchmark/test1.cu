#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void array_generator(int n, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A[idx] = A + idx + blockDim.x;
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
