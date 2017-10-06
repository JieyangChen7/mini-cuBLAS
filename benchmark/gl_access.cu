#include <iostream>
#include <stdio.h>
#include <climits>
#include <algorithm>
#include <cuda_profiler_api.h>
#define LL 15 * 512 
using namespace std;

__global__ void array_generator(double * A, int iteration, int access_per_iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  int L = gridDim.x * blockDim.x;
  for (int i = 0; i < iteration; i++) {
    double * nextA = A + L * access_per_iter;
    for (int j = 0; j < access_per_iter; j++) {
      *(A + L * j) = (unsigned long long int)( nextA + L * j );
    }
    A = nextA;
  }
}


// Kernel for 2048 threads / sm
// Max register use is: 32
// this version disable unroll
__global__ void global_memory_2048(double * A, int iteration, int access_per_iter,
                              unsigned long long int * dStart, unsigned long long int * dEnd) {
  extern __shared__ double cache[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;

  //volatile clock_t start = 0;
  //volatile clock_t end = 0;
  //volatile unsigned long long sum_time = 0;

  double * a_next1 = A;
  double * a_next2 = A + LL;
  double * a_next3 = A + LL * 2;
  double * a_next4 = A + LL * 3;
  double * a_next5 = A + LL * 4;
  double * a_next6 = A + LL * 5;
  double * a_next7 = A + LL * 6;

  # pragma unroll 1
  for (int i = 0; i < iteration; i++) {
    //start = clock();                                                                                                                      
    a_next1 = (double *)(unsigned long long int) *a_next1;
    a_next2 = (double *)(unsigned long long int) *a_next2;
    
    a_next3 = (double *)(unsigned long long int) *a_next3;
    a_next4 = (double *)(unsigned long long int) *a_next4;
    
    a_next5 = (double *)(unsigned long long int) *a_next5;
    a_next6 = (double *)(unsigned long long int) *a_next6;
    a_next7 = (double *)(unsigned long long int) *a_next7;

    //end = clock(); 
  }
  
  *A += (unsigned long long int)a_next1;
  *A +=  (unsigned long long int)a_next2;
  *A +=  (unsigned long long int)a_next3;
  *A +=  (unsigned long long int)a_next4;
    
  *A +=  (unsigned long long int)a_next5;
  *A +=  (unsigned long long int)a_next6;
  *A +=  (unsigned long long int)a_next7;

}

// Kernel for 1024 threads / sm
// Max register use is 64
// this version disable unroll
__global__ void global_memory_1024(double * A, int iteration, int access_per_iter,
                              unsigned long long int * dStart, unsigned long long int * dEnd) {
  extern __shared__ double cache[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;

  //volatile clock_t start = 0;
  //volatile clock_t end = 0;
  //volatile unsigned long long sum_time = 0;

  double * a_next1 = A;
  double * a_next2 = A + LL;
  double * a_next3 = A + LL * 2;
  double * a_next4 = A + LL * 3;
  double * a_next5 = A + LL * 4;
  double * a_next6 = A + LL * 5;
  double * a_next7 = A + LL * 6;
  double * a_next8 = A + LL * 7;
  
  double * a_next9 = A + LL * 8;
  double * a_next10 = A + LL * 9;
  double * a_next11 = A + LL * 10;
  double * a_next12 = A + LL * 11;
  
  double * a_next13 = A + LL * 12;
  double * a_next14 = A + LL * 13;
  double * a_next15 = A + LL * 14;
  double * a_next16 = A + LL * 15;
  double * a_next17 = A + LL * 16;
  double * a_next18 = A + LL * 17;
  double * a_next19 = A + LL * 18;
  double * a_next20 = A + LL * 19;
  double * a_next21 = A + LL * 20;
  double * a_next22 = A + LL * 21;
  double * a_next23 = A + LL * 22;
   
# pragma unroll 1
  for (int i = 0; i < iteration; i++) {
    //start = clock();                                                                                                                      
    a_next1 = (double *)(unsigned long long int) *a_next1;
    a_next2 = (double *)(unsigned long long int) *a_next2; 
    a_next3 = (double *)(unsigned long long int) *a_next3;
    a_next4 = (double *)(unsigned long long int) *a_next4;
    
    a_next5 = (double *)(unsigned long long int) *a_next5;
    a_next6 = (double *)(unsigned long long int) *a_next6;
    a_next7 = (double *)(unsigned long long int) *a_next7;
    a_next8 = (double *)(unsigned long long int) *a_next8;
    
    a_next9 = (double *)(unsigned long long int) *a_next9;
    a_next10 = (double *)(unsigned long long int) *a_next10;
    a_next11 = (double *)(unsigned long long int) *a_next11;
    a_next12 = (double *)(unsigned long long int) *a_next12;
    
    a_next13 = (double *)(unsigned long long int) *a_next13;
    a_next14 = (double *)(unsigned long long int) *a_next14;
    a_next15 = (double *)(unsigned long long int) *a_next15;
    a_next16 = (double *)(unsigned long long int) *a_next16;
    
    a_next17 = (double *)(unsigned long long int) *a_next17;
    a_next18 = (double *)(unsigned long long int) *a_next18;
    a_next19 = (double *)(unsigned long long int) *a_next19;
    a_next20 = (double *)(unsigned long long int) *a_next20;
    
    a_next21 = (double *)(unsigned long long int) *a_next21;
    a_next22 = (double *)(unsigned long long int) *a_next22;
    a_next23 = (double *)(unsigned long long int) *a_next23;
    
    //end = clock(); 
  }
  
  *A += (unsigned long long int)a_next1;
  *A +=  (unsigned long long int)a_next2;
  *A +=  (unsigned long long int)a_next3;
  *A +=  (unsigned long long int)a_next4;
    
  *A +=  (unsigned long long int)a_next5;
  *A +=  (unsigned long long int)a_next6;
  *A +=  (unsigned long long int)a_next7;
  *A +=  (unsigned long long int)a_next8;
  
  *A +=  (unsigned long long int)a_next9;
  *A +=  (unsigned long long int)a_next10;
  *A +=  (unsigned long long int)a_next11;
  *A +=  (unsigned long long int)a_next12;

  *A +=  (unsigned long long int)a_next13;
  *A +=  (unsigned long long int)a_next14;
  *A +=  (unsigned long long int)a_next15;
  *A +=  (unsigned long long int)a_next16;

  *A +=  (unsigned long long int)a_next17;
  *A +=  (unsigned long long int)a_next18;
  *A +=  (unsigned long long int)a_next19;
  *A +=  (unsigned long long int)a_next20;

  *A +=  (unsigned long long int)a_next21;
  *A +=  (unsigned long long int)a_next22;
  *A +=  (unsigned long long int)a_next23;  
}




// Kernel for 1024 threads / sm
// Max regiter use js 64
// this version let compilter to automatic unroll
__global__ void global_memory_1024_2(double * A, int iteration, int access_per_iter,
                              unsigned long long int * dStart, unsigned long long int * dEnd) {
  extern __shared__ double cache[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;

  //volatile clock_t start = 0;
  //volatile clock_t end = 0;
  //volatile unsigned long long sum_time = 0;

  double * a_next1 = A;
  double * a_next2 = A + LL;
  double * a_next3 = A + LL * 2;
  double * a_next4 = A + LL * 3;

  double * a_next5 = A + LL * 4;
  double * a_next6 = A + LL * 5;
  double * a_next7 = A + LL * 6;
  double * a_next8 = A + LL * 7;

  double * a_next9 = A + LL * 8;
  double * a_next10 = A + LL * 9;
  double * a_next11 = A + LL * 10;
  double * a_next12 = A + LL * 11;

  double * a_next13 = A + LL * 12;
  double * a_next14 = A + LL * 13;

  
  for (int i = 0; i < iteration; i++) {
    //start = clock();                                                                                                                      
    a_next1 = (double *)(unsigned long long int) *a_next1;
    a_next2 = (double *)(unsigned long long int) *a_next2;  
    a_next3 = (double *)(unsigned long long int) *a_next3;
    a_next4 = (double *)(unsigned long long int) *a_next4;
    
    a_next5 = (double *)(unsigned long long int) *a_next5;
    a_next6 = (double *)(unsigned long long int) *a_next6;
    a_next7 = (double *)(unsigned long long int) *a_next7;
    a_next8 = (double *)(unsigned long long int) *a_next8;
    
    a_next9 = (double *)(unsigned long long int) *a_next9;
    a_next10 = (double *)(unsigned long long int) *a_next10;
    a_next11 = (double *)(unsigned long long int) *a_next11;
    a_next12 = (double *)(unsigned long long int) *a_next12;
    
    a_next13 = (double *)(unsigned long long int) *a_next13;
    a_next14 = (double *)(unsigned long long int) *a_next14;

    //end = clock(); 
  }
  
  *A += (unsigned long long int)a_next1;
  *A +=  (unsigned long long int)a_next2;
  *A +=  (unsigned long long int)a_next3;
  *A +=  (unsigned long long int)a_next4;
    
  *A +=  (unsigned long long int)a_next5;
  *A +=  (unsigned long long int)a_next6;
  *A +=  (unsigned long long int)a_next7;
  *A +=  (unsigned long long int)a_next8;
  
  *A +=  (unsigned long long int)a_next9;
  *A +=  (unsigned long long int)a_next10;
  *A +=  (unsigned long long int)a_next11;
  *A +=  (unsigned long long int)a_next12;

  *A +=  (unsigned long long int)a_next13;
  *A +=  (unsigned long long int)a_next14;
}



// Kernel for 512 threads / sm
// Max register use is 128
// this version disable unroll
__global__ void global_memory_512(double * A, int iteration, int access_per_iter,
                              unsigned long long int * dStart, unsigned long long int * dEnd) {
  extern __shared__ double cache[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;

  //volatile clock_t start = 0;
  //volatile clock_t end = 0;
  //volatile unsigned long long sum_time = 0;

  double * a_next1 = A;
  double * a_next2 = A + LL;
  double * a_next3 = A + LL * 2;
  double * a_next4 = A + LL * 3;
  double * a_next5 = A + LL * 4;
  double * a_next6 = A + LL * 5;
  double * a_next7 = A + LL * 6;
  double * a_next8 = A + LL * 7;
  
  double * a_next9 = A + LL * 8;
  double * a_next10 = A + LL * 9;
  double * a_next11 = A + LL * 10;
  double * a_next12 = A + LL * 11;
  
  double * a_next13 = A + LL * 12;
  double * a_next14 = A + LL * 13;
  double * a_next15 = A + LL * 14;
  double * a_next16 = A + LL * 15;
  double * a_next17 = A + LL * 16;
  double * a_next18 = A + LL * 17;
  double * a_next19 = A + LL * 18;
  double * a_next20 = A + LL * 19;
  double * a_next21 = A + LL * 20;
  double * a_next22 = A + LL * 21;
  double * a_next23 = A + LL * 22;
  double * a_next24 = A + LL * 23;
  double * a_next25 = A + LL * 24;
  double * a_next26 = A + LL * 25;
  double * a_next27 = A + LL * 26;
  double * a_next28 = A + LL * 27;
  double * a_next29 = A + LL * 28; 
  double * a_next30 = A + LL * 29;

  double * a_next31 = A + LL * 31;
  double * a_next32 = A + LL * 32;
  double * a_next33 = A + LL * 33; 
  double * a_next34 = A + LL * 34;
  double * a_next35 = A + LL * 35; 
  double * a_next36 = A + LL * 36;

  double * a_next37 = A + LL * 37; 
  double * a_next38 = A + LL * 38;
  double * a_next39 = A + LL * 39; 
  double * a_next40 = A + LL * 40;

  double * a_next41 = A + LL * 41;
  double * a_next42 = A + LL * 42;
  double * a_next43 = A + LL * 43; 
  double * a_next44 = A + LL * 44;
  double * a_next45 = A + LL * 45; 
  double * a_next46 = A + LL * 46;

  double * a_next47 = A + LL * 47; 
  double * a_next48 = A + LL * 48;
  double * a_next49 = A + LL * 49; 
  double * a_next50 = A + LL * 50;

  double * a_next51 = A + LL * 51;
  double * a_next52 = A + LL * 52; 
  double * a_next53 = A + LL * 53;
  double * a_next54 = A + LL * 54;
  double * a_next55 = A + LL * 55;


# pragma unroll 1
  for (int i = 0; i < iteration; i++) {
    //start = clock();                                                                                                                      
    a_next1 = (double *)(unsigned long long int) *a_next1;
    a_next2 = (double *)(unsigned long long int) *a_next2; 
    a_next3 = (double *)(unsigned long long int) *a_next3;
    a_next4 = (double *)(unsigned long long int) *a_next4;
    
    a_next5 = (double *)(unsigned long long int) *a_next5;
    a_next6 = (double *)(unsigned long long int) *a_next6;
    a_next7 = (double *)(unsigned long long int) *a_next7;
    a_next8 = (double *)(unsigned long long int) *a_next8;
    
    a_next9 = (double *)(unsigned long long int) *a_next9;
    a_next10 = (double *)(unsigned long long int) *a_next10;
    a_next11 = (double *)(unsigned long long int) *a_next11;
    a_next12 = (double *)(unsigned long long int) *a_next12;
    
    a_next13 = (double *)(unsigned long long int) *a_next13;
    a_next14 = (double *)(unsigned long long int) *a_next14;
    a_next15 = (double *)(unsigned long long int) *a_next15;
    a_next16 = (double *)(unsigned long long int) *a_next16;
    
    a_next17 = (double *)(unsigned long long int) *a_next17;
    a_next18 = (double *)(unsigned long long int) *a_next18;
    a_next19 = (double *)(unsigned long long int) *a_next19;
    a_next20 = (double *)(unsigned long long int) *a_next20;
    
    a_next21 = (double *)(unsigned long long int) *a_next21;
    a_next22 = (double *)(unsigned long long int) *a_next22;
    a_next23 = (double *)(unsigned long long int) *a_next23;
    a_next24 = (double *)(unsigned long long int) *a_next24;

    a_next25 = (double *)(unsigned long long int) *a_next25;
    a_next26 = (double *)(unsigned long long int) *a_next26;
    a_next27 = (double *)(unsigned long long int) *a_next27;
    a_next28 = (double *)(unsigned long long int) *a_next28;

    a_next29 = (double *)(unsigned long long int) *a_next29;
    a_next30 = (double *)(unsigned long long int) *a_next30;
    a_next31 = (double *)(unsigned long long int) *a_next31;
    a_next32 = (double *)(unsigned long long int) *a_next32;

    a_next33 = (double *)(unsigned long long int) *a_next33;
    a_next34 = (double *)(unsigned long long int) *a_next34;
    a_next35 = (double *)(unsigned long long int) *a_next35;
    a_next36 = (double *)(unsigned long long int) *a_next36;

    a_next37 = (double *)(unsigned long long int) *a_next37;
    a_next38 = (double *)(unsigned long long int) *a_next38;
    a_next39 = (double *)(unsigned long long int) *a_next39;
    a_next40 = (double *)(unsigned long long int) *a_next40;

    a_next41 = (double *)(unsigned long long int) *a_next41;
    a_next42 = (double *)(unsigned long long int) *a_next42;
    a_next43 = (double *)(unsigned long long int) *a_next43;
    a_next44 = (double *)(unsigned long long int) *a_next44;
    
    a_next45 = (double *)(unsigned long long int) *a_next45;
    a_next46 = (double *)(unsigned long long int) *a_next46;
    a_next47 = (double *)(unsigned long long int) *a_next47;
    a_next48 = (double *)(unsigned long long int) *a_next48;
    
    a_next49 = (double *)(unsigned long long int) *a_next49;
    a_next50 = (double *)(unsigned long long int) *a_next50;
    a_next51 = (double *)(unsigned long long int) *a_next51;
    a_next52 = (double *)(unsigned long long int) *a_next52;
    
    a_next53 = (double *)(unsigned long long int) *a_next53;
    a_next54 = (double *)(unsigned long long int) *a_next54;
    a_next55 = (double *)(unsigned long long int) *a_next55;
    
    //end = clock(); 
  }
  
  *A += (unsigned long long int)a_next1;
  *A +=  (unsigned long long int)a_next2;
  *A +=  (unsigned long long int)a_next3;
  *A +=  (unsigned long long int)a_next4;
    
  *A +=  (unsigned long long int)a_next5;
  *A +=  (unsigned long long int)a_next6;
  *A +=  (unsigned long long int)a_next7;
  *A +=  (unsigned long long int)a_next8;
  
  *A +=  (unsigned long long int)a_next9;
  *A +=  (unsigned long long int)a_next10;
  *A +=  (unsigned long long int)a_next11;
  *A +=  (unsigned long long int)a_next12;

  *A +=  (unsigned long long int)a_next13;
  *A +=  (unsigned long long int)a_next14;
  *A +=  (unsigned long long int)a_next15;
  *A +=  (unsigned long long int)a_next16;

  *A +=  (unsigned long long int)a_next17;
  *A +=  (unsigned long long int)a_next18;
  *A +=  (unsigned long long int)a_next19;
  *A +=  (unsigned long long int)a_next20;

  *A +=  (unsigned long long int)a_next21;
  *A +=  (unsigned long long int)a_next22;
  *A +=  (unsigned long long int)a_next23;
  *A +=  (unsigned long long int)a_next24;

  *A +=  (unsigned long long int)a_next25;
  *A +=  (unsigned long long int)a_next26;
  *A +=  (unsigned long long int)a_next27;
  *A +=  (unsigned long long int)a_next28;

  *A +=  (unsigned long long int)a_next29;
  *A +=  (unsigned long long int)a_next30;
  *A +=  (unsigned long long int)a_next31;
  *A +=  (unsigned long long int)a_next32;

  *A +=  (unsigned long long int)a_next33;
  *A +=  (unsigned long long int)a_next34;
  *A +=  (unsigned long long int)a_next35;
  *A +=  (unsigned long long int)a_next36;

  *A +=  (unsigned long long int)a_next37;
  *A +=  (unsigned long long int)a_next38;
  *A +=  (unsigned long long int)a_next39;
  *A +=  (unsigned long long int)a_next40;

  *A +=  (unsigned long long int)a_next41;
  *A +=  (unsigned long long int)a_next42;
  *A +=  (unsigned long long int)a_next43;
  *A +=  (unsigned long long int)a_next44;

  *A +=  (unsigned long long int)a_next45;
  *A +=  (unsigned long long int)a_next46;
  *A +=  (unsigned long long int)a_next47;
  *A +=  (unsigned long long int)a_next48;

  *A +=  (unsigned long long int)a_next49;
  *A +=  (unsigned long long int)a_next50;
  *A +=  (unsigned long long int)a_next51;
  *A +=  (unsigned long long int)a_next52;

  *A +=  (unsigned long long int)a_next53;
  *A +=  (unsigned long long int)a_next54;
  *A +=  (unsigned long long int)a_next55;

}


void test_2048(int block_size){
  int iteration = 1000;
  int access_per_iter = 7;
  int SM = 15;
  int block_per_sm = 2048/block_size;
  int total_block = SM * block_per_sm;
  //int block_size = 1024;
  cout << "Total concurrent threads/SM: " << block_per_sm * block_size << endl;
  int n = total_block * block_size * access_per_iter * (iteration + 1);
  double * A = new double[n];
  unsigned long long int * start = new unsigned long long int[n];
  unsigned long long int * end = new unsigned long long int[n];
  unsigned long long int * dStart;
  unsigned long long int * dEnd;
  double * dA;
  cudaMalloc(&dA, (n) * sizeof(double));
  cudaMalloc((void**)&dStart, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dEnd, n * sizeof(unsigned long long int));

  array_generator<<<total_block, block_size>>>(dA, iteration, access_per_iter);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<array_gene>Error: %s\n", cudaGetErrorString(err));

  clock_t t = clock();
  global_memory_2048<<<total_block, block_size, 49152 / block_per_sm>>>(dA, iteration, access_per_iter, dStart, dEnd);
  cudaDeviceSynchronize();
  t = clock() - t;

  float real_time = ((float)t)/CLOCKS_PER_SEC;
  cout <<"Runing time: " << real_time << " s." << endl;
  long long total_byte = total_block * block_size * sizeof(double) * access_per_iter;
  double total_gb = total_byte/1e9;
  total_gb *= iteration;
  cout << "Total data requested:"<<total_gb << " GB."<< endl;
  double throughput = total_gb/real_time;
  cout <<"Throughput: " << throughput << " GB/s." << endl;
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<global_memory>Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(A, dA, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dStart);
  cudaFree(dEnd);
  delete [] A;
  delete [] start;
  delete [] end;  

}

void test_1024(int block_size){
  int iteration = 1000;
  int access_per_iter = 23;
  int SM = 15;
  int block_per_sm = 1024/block_size;
  int total_block = SM * block_per_sm;
  //int block_size = 1024;

  int n = total_block * block_size * access_per_iter * (iteration + 1);
  double * A = new double[n];
  unsigned long long int * start = new unsigned long long int[n];
  unsigned long long int * end = new unsigned long long int[n];
  unsigned long long int * dStart;
  unsigned long long int * dEnd;
  double * dA;
  cudaMalloc(&dA, (n) * sizeof(double));
  cudaMalloc((void**)&dStart, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dEnd, n * sizeof(unsigned long long int));

  array_generator<<<total_block, block_size>>>(dA, iteration, access_per_iter);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<array_gene>Error: %s\n", cudaGetErrorString(err));

  clock_t t = clock();
  global_memory_1024<<<total_block, block_size, 49152 / block_per_sm>>>(dA, iteration, access_per_iter, dStart, dEnd);
  cudaDeviceSynchronize();
  t = clock() - t;

  float real_time = ((float)t)/CLOCKS_PER_SEC;
  cout <<"Runing time: " << real_time << " s." << endl;
  long long total_byte = total_block * block_size * sizeof(double) * access_per_iter;
  double total_gb = total_byte/1e9;
  total_gb *= iteration;
  cout << "Total data requested:"<<total_gb << " GB."<< endl;
  double throughput = total_gb/real_time;
  cout <<"Throughput: " << throughput << " GB/s." << endl;
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<global_memory>Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(A, dA, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dStart);
  cudaFree(dEnd);
  delete [] A;
  delete [] start;
  delete [] end;  
}


void test_512(int block_size){
  int iteration = 1000;
  int access_per_iter = 34;
  int SM = 15;
  int block_per_sm = 512/block_size;
  int total_block = SM * block_per_sm;
  //int block_size = 1024;

  int n = total_block * block_size * access_per_iter * (iteration + 1);
  double * A = new double[n];
  unsigned long long int * start = new unsigned long long int[n];
  unsigned long long int * end = new unsigned long long int[n];
  unsigned long long int * dStart;
  unsigned long long int * dEnd;
  double * dA;
  cudaMalloc(&dA, (n) * sizeof(double));
  cudaMalloc((void**)&dStart, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dEnd, n * sizeof(unsigned long long int));

  array_generator<<<total_block, block_size>>>(dA, iteration, access_per_iter);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<array_gene>Error: %s\n", cudaGetErrorString(err));

  clock_t t = clock();
  global_memory_512<<<total_block, block_size, 49152 / block_per_sm>>>(dA, iteration, access_per_iter, dStart, dEnd);
  cudaDeviceSynchronize();
  t = clock() - t;

  float real_time = ((float)t)/CLOCKS_PER_SEC;
  cout <<"Runing time: " << real_time << " s." << endl;
  long long total_byte = total_block * block_size * sizeof(double) * access_per_iter;
  double total_gb = total_byte/1e9;
  total_gb *= iteration;
  cout << "Total data requested:"<<total_gb << " GB."<< endl;
  double throughput = total_gb/real_time;
  cout <<"Throughput: " << throughput << " GB/s." << endl;
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<global_memory>Error: %s\n", cudaGetErrorString(err));

  cudaMemcpy(A, dA, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  cudaFree(dStart);
  cudaFree(dEnd);
  delete [] A;
  delete [] start;
  delete [] end;  
}


int main(){
  if (LL / 15 == 1024) { 
    for (int i = 64; i <= 1024; i *= 2) {
      cout << "block size: " << i << endl;
      test_1024(i);
    }
  } else if (LL / 15 == 2048) {
    for (int i = 128; i <= 1024; i *= 2) {
      cout << "block size: " << i << endl;
      test_2048(i);
    }
  } else if (LL / 15 == 512) {
    for (int i = 32; i <= 1024; i *= 2) {
      cout << "block size: " << i << endl;
      test_512(i);
    }
  }
  

}
