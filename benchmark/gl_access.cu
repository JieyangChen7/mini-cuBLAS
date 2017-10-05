#include <iostream>
#include <stdio.h>
#include <climits>
#include <algorithm>
using namespace std

__global__ void array_generator(double * A, int iteration, int access_per_iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  idx = idx * access_per_iter;
  A = A + idx;
  for (int i = 0; i < iteration; i++) {
    for (int j = 0 ; j < access_per_iter; j++) {
      A[j] = (unsigned long long int)(A + j + gridDim.x * blockDim.x * access_per_iter );
      //if (idx == 0) {                                                                                                                     
      //        printf("%d-%d-%d\n",i,j, (unsigned long long int)(A + j + gridDim.x * blockDim.x * access_per_iter ));                      
      //}                                                                                                                                   
    }
    A += gridDim.x * blockDim.x * access_per_iter;
  }

}


__global__ void global_memory(double * A, int iteration, int access_per_iter,
                              unsigned long long int * dStart, unsigned long long int * dEnd) {
  extern __shared__ double cache[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  idx = idx * access_per_iter;
  A = A + idx;

  volatile clock_t start = 0;
  volatile clock_t end = 0;
  volatile unsigned long long sum_time = 0;

  double * a_next1;
  double * a_next2;
  double * a_next3;
  double * a_next4;
  double * a_next5;
  double * a_next6;
  double * a_next7;
  double * a_next8;
  double * a_next9;
  double * a_next10;
  double * a_next11;
  double * a_next12;
  double * a_next13;
  double * a_next14;
  double * a_next15;
  double * a_next16;


  double * a_curr1 = A;
  double * a_curr2 = A + 1;
  double * a_curr3 = A + 2;
  double * a_curr4 = A + 3;
  double * a_curr5 = A + 4;
  double * a_curr6 = A + 5;
  double * a_curr7 = A + 6;
  double * a_curr8 = A + 7;
  double * a_curr9 = A + 8;
  double * a_curr10 = A + 9;
  double * a_curr11 = A + 10;
  double * a_curr12 = A + 11;
  double * a_curr13 = A + 12;
  double * a_curr14 = A + 13;
  double * a_curr15 = A + 14;
  double * a_curr16 = A + 15;

  for (int i = 0; i < iteration; i++) {
    //start = clock();                                                                                                                      
    a_next1 = (double *)(unsigned long long int) *a_curr1;
    a_next2 = (double *)(unsigned long long int) *a_curr2;
    a_next3 = (double *)(unsigned long long int) *a_curr3;
    a_next4 = (double *)(unsigned long long int) *a_curr4;
    a_next5 = (double *)(unsigned long long int) *a_curr5;
    a_next6 = (double *)(unsigned long long int) *a_curr6;
    a_next7 = (double *)(unsigned long long int) *a_curr7;
    a_next8 = (double *)(unsigned long long int) *a_curr8;
    a_next9 = (double *)(unsigned long long int) *a_curr9;
    a_next10 = (double *)(unsigned long long int) *a_curr10;
    a_next11 = (double *)(unsigned long long int) *a_curr11;
    a_next12 = (double *)(unsigned long long int) *a_curr12;
    a_next13 = (double *)(unsigned long long int) *a_curr13;
    a_next14 = (double *)(unsigned long long int) *a_curr14;
    a_next15 = (double *)(unsigned long long int) *a_curr15;
    a_next16 = (double *)(unsigned long long int) *a_curr16;
    //if (idx == 0)                                                                                                                         
    //  printf("%d-%d-%d\n", i, 0, (unsigned long long int) *a_curr1);   
    __syncthreads();
    a_curr1 = a_next1;
    a_curr2 = a_next2;
    a_curr3 = a_next3;
    a_curr4 = a_next4;
    a_curr5 = a_next5;
    a_curr6 = a_next6;
    a_curr7 = a_next7;
    a_curr8 = a_next8;
    a_curr9 = a_next9;
    a_curr10 = a_next10;
    a_curr11 = a_next11;
    a_curr12 = a_next12;
    a_curr13 = a_next13;
    a_curr14 = a_next14;
    a_curr15 = a_next15;
    a_curr16 = a_next16;
    //end = clock(); 
  }

  *A = (unsigned long long int)a_curr1 +
    (unsigned long long int)a_curr2 +
    (unsigned long long int)a_curr3 +
    (unsigned long long int)a_curr4 +
    (unsigned long long int)a_curr5 +
    (unsigned long long int)a_curr6 +
    (unsigned long long int)a_curr7 +
    (unsigned long long int)a_curr8 +
    (unsigned long long int)a_curr9 +
    (unsigned long long int)a_curr10 +
    (unsigned long long int)a_curr11 +
    (unsigned long long int)a_curr12 +
    (unsigned long long int)a_curr13 +
    (unsigned long long int)a_curr14 +
    (unsigned long long int)a_curr15 +
    (unsigned long long int)a_curr16 ;
}



int main(){

  int SM = 30;
  int B = 1024;
  //int n = SM*B*32;                                                                                                                        
  int n = SM*B*16*100;
  double * A = new double[n];
  unsigned long long int * start = new unsigned long long int[n];
  unsigned long long int * end = new unsigned long long int[n];
  unsigned long long int *dStart;
  unsigned long long int *dEnd;
  cudaMalloc(&dA, (n) * sizeof(double));
  cudaMalloc((void**)&dStart, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dEnd, n * sizeof(unsigned long long int));



  int iteration = 50;
  int access_per_iter = 16;
  array_generator<<<SM, B>>>(dA, iteration, access_per_iter);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<array_gene>Error: %s\n", cudaGetErrorString(err));

  clock_t t = clock();
  global_memory<<<SM, B, 49152>>>(dA, iteration, access_per_iter, dStart, dEnd);
  cudaDeviceSynchronize();
  t = clock() - t;

  float real_time = ((float)t)/CLOCKS_PER_SEC;
  cout <<"Runing time: " << real_time << " s." << endl;
  long long total_byte = SM*B*64*access_per_iter;
  double total_gb = total_byte/1e9;
  total_gb *= iteration;
  cout << "Total data requested:"<<total_gb << " GB."<< endl;
  double throughput = total_gb/real_time;
  cout <<"Throughput: " << throughput << " GB/s." << endl;
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<global_memory>Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(A, dA, (n + B*10) * sizeof(double), cudaMemcpyDeviceToHost);

}
