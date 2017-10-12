#include <iostream>
#include <stdio.h>
#include <climits>
#include <algorithm>
#include <cuda_profiler_api.h>
#define SM 24
#define LL SM * 2048 
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
__global__ void global_memory_2048(double * A, int iteration, int access_per_iter) {
  extern __shared__ double cache[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;

  //volatile clock_t start = 0;
  //volatile clock_t end = 0;
  //volatile unsigned long long sum_time = 0;

   register double * a_next1 = A;
   register double * a_next2 = A + LL;
   register double * a_next3 = A + LL * 2;
   register double * a_next4 = A + LL * 3;
   register double * a_next5 = A + LL * 4;
   register double * a_next6 = A + LL * 5;
   register double * a_next7 = A + LL * 6;
  //register double * a_next8 = A + LL * 7;

   register double temp = 0;
  
  # pragma unroll 1 
  for (int i = 0; i < iteration; i++) {
    //start = clock(); 

   
    a_next1 = (double *)(unsigned long long int) *a_next1;
    // a_next2 = (double *)(unsigned long long int) *a_next2;
    
    // a_next3 = (double *)(unsigned long long int) *a_next3;
    // a_next4 = (double *)(unsigned long long int) *a_next4;
    

    // a_next5 = (double *)(unsigned long long int) *a_next5;
    // a_next6 = (double *)(unsigned long long int) *a_next6;
    // a_next7 = (double *)(unsigned long long int) *a_next7;
  // a_next8 = (double *)(unsigned long long int) *a_next8;

     temp += temp * iteration;
     //temp += temp *iteration;
    // temp += temp *iteration;
    // temp += temp *iteration;

    // temp += temp * iteration;
    // temp += temp *iteration;
    // temp += temp *iteration;
    // temp += temp *iteration;

    // temp += temp * iteration;
    // temp += temp *iteration;
    // temp += temp *iteration;
    // temp += temp *iteration;

    // temp += temp * iteration;
    // temp += temp *iteration;
    // temp += temp *iteration;
    // temp += temp *iteration;


    //__syncthreads();
    
    //end = clock(); 
  }
  
  *A += (unsigned long long int)a_next1;
  // *A +=  (unsigned long long int)a_next2;
  // *A +=  (unsigned long long int)a_next3;
  // *A +=  (unsigned long long int)a_next4;
   *A += temp;
  // *A +=  (unsigned long long int)a_next5;
  // *A +=  (unsigned long long int)a_next6;
  // *A +=  (unsigned long long int)a_next7;
 // *A +=  (unsigned long long int)a_next8;

}


void test_2048(int block_size){
  int iteration = 1000;
  int access_per_iter = 1;
  int compute_per_iter = 2;
  //int SM = 24;
  int block_per_sm = 2048/block_size;
  int total_block = SM * block_per_sm;
  //int block_size = 1024;
  cout << "Total concurrent threads/SM: " << block_per_sm * block_size << endl;
  cout << "Total block: " << total_block << endl;
  int n = total_block * block_size * access_per_iter * (iteration + 1);
  double * A = new double[n];
  unsigned long long int * start = new unsigned long long int[n];
  unsigned long long int * end = new unsigned long long int[n];
  //unsigned long long int * dStart;
  //unsigned long long int * dEnd;
  double * dA;
  cudaMalloc(&dA, (n) * sizeof(double));
  //cudaMalloc((void**)&dStart, n * sizeof(unsigned long long int));
  //cudaMalloc((void**)&dEnd, n * sizeof(unsigned long long int));

  array_generator<<<total_block, block_size>>>(dA, iteration, access_per_iter);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<array_gene>Error: %s\n", cudaGetErrorString(err));

  cudaEvent_t t1, t2;
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  cudaEventRecord(t1);
  //clock_t t = clock();
  global_memory_2048<<<total_block, block_size>>>(dA, iteration, access_per_iter);
  cudaEventRecord(t2);

  cudaEventSynchronize(t2);
  //cudaDeviceSynchronize();
  //t = clock() - t;

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, t1, t2);
  double real_time = milliseconds/1000;
  //double real_time = ((double)t)/CLOCKS_PER_SEC;
  cout <<"Runing time: " << real_time << " s." << endl;
  long long total_byte = total_block * block_size * sizeof(double) * access_per_iter;
  double total_gb = (double)total_byte/1e9;
  total_gb *= iteration;
  cout << "Total data requested:"<<total_gb << " GB."<< endl;
  double throughput = total_gb/real_time;
  cout <<"Throughput: " << throughput << " GB/s." << endl;
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("<global_memory>Error: %s\n", cudaGetErrorString(err));

  long long total_ops = total_block * block_size * access_per_iter * compute_per_iter;
  double perf = (double)total_ops/(real_time * 1e9);
  cout <<"Perf: " << perf << " Gflop/s." << endl;


  cudaMemcpy(A, dA, n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dA);
  //cudaFree(dStart);
  //cudaFree(dEnd);
  delete [] A;
  delete [] start;
  delete [] end;  

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

  double * a_next31 = A + LL * 30;
  double * a_next32 = A + LL * 31;
  double * a_next33 = A + LL * 32; 
  double * a_next34 = A + LL * 33;
  double * a_next35 = A + LL * 34; 
  double * a_next36 = A + LL * 35;

  double * a_next37 = A + LL * 36; 
  double * a_next38 = A + LL * 37;
  double * a_next39 = A + LL * 38; 
  double * a_next40 = A + LL * 39;

  double * a_next41 = A + LL * 40;
  double * a_next42 = A + LL * 41;
  double * a_next43 = A + LL * 42; 
  double * a_next44 = A + LL * 43;
  double * a_next45 = A + LL * 44; 
  double * a_next46 = A + LL * 45;

  double * a_next47 = A + LL * 46; 
  double * a_next48 = A + LL * 47;
  double * a_next49 = A + LL * 48; 
  double * a_next50 = A + LL * 49;

  double * a_next51 = A + LL * 50;
  double * a_next52 = A + LL * 51; 
  double * a_next53 = A + LL * 52;
  double * a_next54 = A + LL * 53;
  double * a_next55 = A + LL * 54;


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


// Kernel for 256 threads / sm
// Max register use is 256
// this version disable unroll
__global__ void global_memory_256(double * A, int iteration, int access_per_iter,
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
  double * a_next31 = A + LL * 30;
  double * a_next32 = A + LL * 31;

  double * a_next33 = A + LL * 32; 
  double * a_next34 = A + LL * 33;
  double * a_next35 = A + LL * 34; 
  double * a_next36 = A + LL * 35;

  double * a_next37 = A + LL * 36; 
  double * a_next38 = A + LL * 37;
  double * a_next39 = A + LL * 38; 
  double * a_next40 = A + LL * 39;

  double * a_next41 = A + LL * 40;
  double * a_next42 = A + LL * 41;
  double * a_next43 = A + LL * 42; 
  double * a_next44 = A + LL * 43;

  double * a_next45 = A + LL * 44; 
  double * a_next46 = A + LL * 45;
  double * a_next47 = A + LL * 46; 
  double * a_next48 = A + LL * 47;

  double * a_next49 = A + LL * 48; 
  double * a_next50 = A + LL * 49;
  double * a_next51 = A + LL * 50;
  double * a_next52 = A + LL * 51;

  double * a_next53 = A + LL * 52;
  double * a_next54 = A + LL * 53;
  double * a_next55 = A + LL * 54;
  double * a_next56 = A + LL * 55;

  double * a_next57 = A + LL * 56; 
  double * a_next58 = A + LL * 57;
  double * a_next59 = A + LL * 58; 
  double * a_next60 = A + LL * 59;

  double * a_next61 = A + LL * 60;
  double * a_next62 = A + LL * 61;
  double * a_next63 = A + LL * 62;
  double * a_next64 = A + LL * 63;

  double * a_next65 = A + LL * 64;
  double * a_next66 = A + LL * 65;
  double * a_next67 = A + LL * 66; 
  double * a_next68 = A + LL * 67;

  double * a_next69 = A + LL * 68; 
  double * a_next70 = A + LL * 69;
  double * a_next71 = A + LL * 70;
  double * a_next72 = A + LL * 71;

  double * a_next73 = A + LL * 72;
  double * a_next74 = A + LL * 73;
  double * a_next75 = A + LL * 74;
  double * a_next76 = A + LL * 75;

  double * a_next77 = A + LL * 76;
  double * a_next78 = A + LL * 77; 
  double * a_next79 = A + LL * 78;
  double * a_next80 = A + LL * 79;

  double * a_next81 = A + LL * 80;
  double * a_next82 = A + LL * 81;
  double * a_next83 = A + LL * 82;
  double * a_next84 = A + LL * 83;

  double * a_next85 = A + LL * 84;
  double * a_next86 = A + LL * 85;
  double * a_next87 = A + LL * 86; 
  double * a_next88 = A + LL * 87;

  double * a_next89 = A + LL * 88; 
  double * a_next90 = A + LL * 89;
  double * a_next91 = A + LL * 90;
  double * a_next92 = A + LL * 91;

  double * a_next93 = A + LL * 92;
  double * a_next94 = A + LL * 93;
  double * a_next95 = A + LL * 94;
  double * a_next96 = A + LL * 95;
  
  double * a_next97 = A + LL * 96;
  double * a_next98 = A + LL * 97; 
  double * a_next99 = A + LL * 98;
  double * a_next100 = A + LL * 99;

  double * a_next101 = A + LL * 100;
  double * a_next102 = A + LL * 101;
  double * a_next103 = A + LL * 102;
  double * a_next104 = A + LL * 103;

  double * a_next105 = A + LL * 104;
  double * a_next106 = A + LL * 105;
  double * a_next107 = A + LL * 106; 
  double * a_next108 = A + LL * 107;

  double * a_next109 = A + LL * 108; 
  double * a_next110 = A + LL * 109;
  double * a_next111 = A + LL * 110;
  double * a_next112 = A + LL * 111;

  double * a_next113 = A + LL * 112;
  double * a_next114 = A + LL * 113;
  double * a_next115 = A + LL * 114;
  double * a_next116 = A + LL * 115;
  
  double * a_next117 = A + LL * 116;
  double * a_next118 = A + LL * 117; 
  double * a_next119 = A + LL * 118;
  double * a_next120 = A + LL * 119;


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
    a_next56 = (double *)(unsigned long long int) *a_next56;

    a_next57 = (double *)(unsigned long long int) *a_next57;
    a_next58 = (double *)(unsigned long long int) *a_next58;
    a_next59 = (double *)(unsigned long long int) *a_next59;
    a_next60 = (double *)(unsigned long long int) *a_next60;

    a_next61 = (double *)(unsigned long long int) *a_next61;
    a_next62 = (double *)(unsigned long long int) *a_next62;
    a_next63 = (double *)(unsigned long long int) *a_next63;
    a_next64 = (double *)(unsigned long long int) *a_next64;

    a_next65 = (double *)(unsigned long long int) *a_next65;
    a_next66 = (double *)(unsigned long long int) *a_next66;
    a_next67 = (double *)(unsigned long long int) *a_next67;
    a_next68 = (double *)(unsigned long long int) *a_next68;
    
    a_next69 = (double *)(unsigned long long int) *a_next69;
    a_next70 = (double *)(unsigned long long int) *a_next70;
    a_next71 = (double *)(unsigned long long int) *a_next71;
    a_next72 = (double *)(unsigned long long int) *a_next72;
    
    a_next73 = (double *)(unsigned long long int) *a_next73;
    a_next74 = (double *)(unsigned long long int) *a_next74;
    a_next75 = (double *)(unsigned long long int) *a_next75;
    a_next76 = (double *)(unsigned long long int) *a_next76;

    a_next77 = (double *)(unsigned long long int) *a_next77;
    a_next78 = (double *)(unsigned long long int) *a_next78;
    a_next79 = (double *)(unsigned long long int) *a_next79;
    a_next80 = (double *)(unsigned long long int) *a_next80;

    a_next81 = (double *)(unsigned long long int) *a_next81;
    a_next82 = (double *)(unsigned long long int) *a_next82;
    a_next83 = (double *)(unsigned long long int) *a_next83;
    a_next84 = (double *)(unsigned long long int) *a_next84;

    a_next85 = (double *)(unsigned long long int) *a_next85;
    a_next86 = (double *)(unsigned long long int) *a_next86;
    a_next87 = (double *)(unsigned long long int) *a_next87;
    a_next88 = (double *)(unsigned long long int) *a_next88;
  
    a_next89 = (double *)(unsigned long long int) *a_next89;
    a_next90 = (double *)(unsigned long long int) *a_next90;
    a_next91 = (double *)(unsigned long long int) *a_next91;
    a_next92 = (double *)(unsigned long long int) *a_next92;
    
    a_next93 = (double *)(unsigned long long int) *a_next93;
    a_next94 = (double *)(unsigned long long int) *a_next94;
    a_next95 = (double *)(unsigned long long int) *a_next95;
    a_next96 = (double *)(unsigned long long int) *a_next96;

    a_next97 = (double *)(unsigned long long int) *a_next97;
    a_next98 = (double *)(unsigned long long int) *a_next98;
    a_next99 = (double *)(unsigned long long int) *a_next99;
    a_next100 = (double *)(unsigned long long int) *a_next100;

    a_next101 = (double *)(unsigned long long int) *a_next101;
    a_next102 = (double *)(unsigned long long int) *a_next102;
    a_next103 = (double *)(unsigned long long int) *a_next103;
    a_next104 = (double *)(unsigned long long int) *a_next104;

    a_next105 = (double *)(unsigned long long int) *a_next105;
    a_next106 = (double *)(unsigned long long int) *a_next106;
    a_next107 = (double *)(unsigned long long int) *a_next107;
    a_next108 = (double *)(unsigned long long int) *a_next108;
  
    a_next109 = (double *)(unsigned long long int) *a_next109;
    a_next110 = (double *)(unsigned long long int) *a_next110;
    a_next111 = (double *)(unsigned long long int) *a_next111;
    a_next112 = (double *)(unsigned long long int) *a_next112;
    
    a_next113 = (double *)(unsigned long long int) *a_next113;
    a_next114 = (double *)(unsigned long long int) *a_next114;
    a_next115 = (double *)(unsigned long long int) *a_next115;
    a_next116 = (double *)(unsigned long long int) *a_next116;

    a_next117 = (double *)(unsigned long long int) *a_next117;
    a_next118 = (double *)(unsigned long long int) *a_next118;
    a_next119 = (double *)(unsigned long long int) *a_next119;
    a_next120 = (double *)(unsigned long long int) *a_next120;
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
  *A +=  (unsigned long long int)a_next56;

  *A +=  (unsigned long long int)a_next57;
  *A +=  (unsigned long long int)a_next58;
  *A +=  (unsigned long long int)a_next59;
  *A +=  (unsigned long long int)a_next60;

  *A +=  (unsigned long long int)a_next61;
  *A +=  (unsigned long long int)a_next62;
  *A +=  (unsigned long long int)a_next63;
  *A +=  (unsigned long long int)a_next64;

  *A +=  (unsigned long long int)a_next65;
  *A +=  (unsigned long long int)a_next66;
  *A +=  (unsigned long long int)a_next67;
  *A +=  (unsigned long long int)a_next68;

  *A +=  (unsigned long long int)a_next69;
  *A +=  (unsigned long long int)a_next70;
  *A +=  (unsigned long long int)a_next71;
  *A +=  (unsigned long long int)a_next72;

  *A +=  (unsigned long long int)a_next73;
  *A +=  (unsigned long long int)a_next74;
  *A +=  (unsigned long long int)a_next75;
  *A +=  (unsigned long long int)a_next76;

  *A +=  (unsigned long long int)a_next77;
  *A +=  (unsigned long long int)a_next78;
  *A +=  (unsigned long long int)a_next79;
  *A +=  (unsigned long long int)a_next80;

  *A +=  (unsigned long long int)a_next81;
  *A +=  (unsigned long long int)a_next82;
  *A +=  (unsigned long long int)a_next83;
  *A +=  (unsigned long long int)a_next84;

  *A +=  (unsigned long long int)a_next85;
  *A +=  (unsigned long long int)a_next86;
  *A +=  (unsigned long long int)a_next87;
  *A +=  (unsigned long long int)a_next88;

  *A +=  (unsigned long long int)a_next89;
  *A +=  (unsigned long long int)a_next90;
  *A +=  (unsigned long long int)a_next91;
  *A +=  (unsigned long long int)a_next92;

  *A +=  (unsigned long long int)a_next93;
  *A +=  (unsigned long long int)a_next94;
  *A +=  (unsigned long long int)a_next95;
  *A +=  (unsigned long long int)a_next96;

  *A +=  (unsigned long long int)a_next97;
  *A +=  (unsigned long long int)a_next98;
  *A +=  (unsigned long long int)a_next99;
  *A +=  (unsigned long long int)a_next100;

  *A +=  (unsigned long long int)a_next101;
  *A +=  (unsigned long long int)a_next102;
  *A +=  (unsigned long long int)a_next103;
  *A +=  (unsigned long long int)a_next104;

  *A +=  (unsigned long long int)a_next105;
  *A +=  (unsigned long long int)a_next106;
  *A +=  (unsigned long long int)a_next107;
  *A +=  (unsigned long long int)a_next108;

  *A +=  (unsigned long long int)a_next109;
  *A +=  (unsigned long long int)a_next110;
  *A +=  (unsigned long long int)a_next111;
  *A +=  (unsigned long long int)a_next112;

  *A +=  (unsigned long long int)a_next113;
  *A +=  (unsigned long long int)a_next114;
  *A +=  (unsigned long long int)a_next115;
  *A +=  (unsigned long long int)a_next116;

  *A +=  (unsigned long long int)a_next117;
  *A +=  (unsigned long long int)a_next118;
  *A +=  (unsigned long long int)a_next119;
  *A +=  (unsigned long long int)a_next120;

}




void test_1024(int block_size){
  int iteration = 100;
  int access_per_iter = 23;
  //int SM = 15;
  int block_per_sm = 1024/block_size;
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

  cudaEvent_t t1, t2;
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  cudaEventRecord(t1);
  global_memory_1024<<<total_block, block_size>>>(dA, iteration, access_per_iter, dStart, dEnd);
  cudaEventRecord(t2);

  cudaEventSynchronize(t2);
  //cudaDeviceSynchronize();
  //t = clock() - t;

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, t1, t2);
  double real_time = milliseconds/1000;

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
  int iteration = 100;
  int access_per_iter = 55;
  //int SM = 15;
  int block_per_sm = 512/block_size;
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

  cudaEvent_t t1, t2;
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  cudaEventRecord(t1);
  global_memory_512<<<total_block, block_size>>>(dA, iteration, access_per_iter, dStart, dEnd);
  cudaEventRecord(t2);

  cudaEventSynchronize(t2);
  //cudaDeviceSynchronize();
  //t = clock() - t;

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, t1, t2);
  double real_time = milliseconds/1000;

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


void test_256(int block_size){
  int iteration = 100;
  int access_per_iter = 120;
  //int SM = 15;
  int block_per_sm = 256/block_size;
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

  cudaEvent_t t1, t2;
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  cudaEventRecord(t1);
  global_memory_256<<<total_block, block_size, 49152 / block_per_sm>>>(dA, iteration, access_per_iter, dStart, dEnd);
  cudaEventRecord(t2);

  cudaEventSynchronize(t2);
  //cudaDeviceSynchronize();
  //t = clock() - t;

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, t1, t2);
  double real_time = milliseconds/1000;

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
  cout << "start benchmark" << endl;
  if (LL / SM == 1024) { 
    for (int i = 64; i <= 1024; i *= 2) {
      cout << "block size: " << i << endl;
      test_1024(i);
    }
  } else if (LL / SM == 2048) {
    for (int i = 128; i <= 1024; i *= 2) {
      cout << "block size: " << i << endl;
      test_2048(i);
    }
  } else if (LL / SM == 512) {
    for (int i = 32; i <= 512; i *= 2) {
      cout << "block size: " << i << endl;
      test_512(i);
    }
  } else if (LL / SM == 256) {
    for (int i = 16; i <= 256; i *= 2) {
      cout << "block size: " << i << endl;
      test_256(i);
    }
  }
  

}
