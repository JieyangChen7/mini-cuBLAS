#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void array_generator(int n, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A[idx] = A + idx + blockDim.x;
}


int main(){
  int n = 1024;
  int B = 128;
  double * A = new double[n + B];
  double * dA;
  cudaMalloc(&dA, (n + B) * sizeof(double));

}
