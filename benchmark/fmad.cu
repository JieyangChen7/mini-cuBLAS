#include <iostream>
#include <stdio.h>
#include <climits>
#include <algorithm>
#include <cuda_profiler_api.h>
#define SM 24
#define LL SM * 2048 
using namespace std;


// Kernel for 2048 threads / sm
// Max register use is: 32
// this version disable unroll
__global__ void global_memory_2048(double * A, int iteration, int access_per_iter, unsigned long long int * time) {
  extern __shared__ double cache[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;

  volatile unsigned long long int start = 0;
  volatile unsigned long long int end = 0;
  //volatile unsigned long long sum_time = 0;

  

  register double temp = 0;
  start = clock(); 
  # pragma unroll 1 
  for (int i = 0; i < iteration; i++) {
    

     temp += temp * iteration;
    temp += temp *iteration;
    temp += temp *iteration;
    temp += temp *iteration;

    temp += temp * iteration;
    temp += temp *iteration;
    temp += temp *iteration;
    temp += temp *iteration;

    temp += temp * iteration;
    temp += temp *iteration;
    temp += temp *iteration;
    temp += temp *iteration;

    temp += temp * iteration;
    temp += temp *iteration;
    temp += temp *iteration;
    temp += temp *iteration;

  }
  end = clock();
  *A += temp;

 // *A +=  (unsigned long long int)a_next8;

}



void test_2048(int block_size){
  int iteration = 1000;
  int access_per_iter = 16;
  //int SM = 24;
  int block_per_sm = 2048/block_size;
  int total_block = SM * block_per_sm;
  //int block_size = 1024;
  cout << "Total concurrent threads/SM: " << block_per_sm * block_size << endl;
  cout << "Total block: " << total_block << endl;
  int n = total_block * block_size;
  double * A = new double[n];
  unsigned long long int * start = new unsigned long long int[n];
  unsigned long long int * end = new unsigned long long int[n];
  unsigned long long int * dStart;
  unsigned long long int * dEnd;
  double * dA;
  cudaMalloc(&dA, (n) * sizeof(double));
  cudaMalloc((void**)&dStart, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dEnd, n * sizeof(unsigned long long int));

  

  cudaEvent_t t1, t2;
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  cudaEventRecord(t1);
  //clock_t t = clock();
  global_memory_2048<<<total_block, block_size>>>(dA, iteration, access_per_iter, dStart);
  cudaEventRecord(t2);

  cudaEventSynchronize(t2);
  //cudaDeviceSynchronize();
  //t = clock() - t;

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, t1, t2);
  double real_time = milliseconds/1000;
  //double real_time = ((double)t)/CLOCKS_PER_SEC;
  cout <<"Runing time: " << real_time << " s." << endl;
  long long total_ops = total_block * block_size * access_per_iter;
  double total_gb = (double)total_ops/1e9;
  total_gb *= iteration;
  cout << "Total ops:"<<total_gb << " Gflos."<< endl;
  double throughput = total_gb/real_time;
  cout <<"Perf: " << throughput << " Gflop/s." << endl;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<global_memory>Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(A, dA, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(start, dStart, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);

  for (int i = 0 ; i < block_per_sm * block_size ; i++) {
    cout << "[" << i << "]: " << start[i];
  }

  //cudaFree(dStart);
  //cudaFree(dEnd);
  delete [] A;
  delete [] start;
  delete [] end;  

}



int main(){
  cout << "start benchmark" << endl;
  
    for (int i = 1024; i <= 1024; i *= 2) {
      cout << "block size: " << i << endl;
      test_2048(i);
    }


}
