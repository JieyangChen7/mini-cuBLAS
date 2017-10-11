#include <stdlib.h>
#include <iostream>
#include "cublas_v2.h"
#include <cmath>
#include <time.h>
#include <stdio.h>
#define TEST_RUN 10 
#define ESP 10e-10
using namespace std;


void check_cuda_error(){
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
}

void check_C(double * dC, int m, int n, double * checkC) {
  for (int i = 0; i < m * n; i++){
    //cout << i << endl;
    if (fabs(dC[i] - checkC[i]) > ESP){
      cout << "error:" << fabs(dC[i] - checkC[i]) << endl;
      return;
    }
  }
  cout << "correct" << endl;
}

/////////////////////////NAIVE/////////////////////////
__global__ void
dgemm_kernel_naive(int m, int n, int k, double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  //determine the row to process                                                        
  register int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  register double temp1 = 0;
  register double temp2 = 0;
  register double a = 0;
  register double b1 = 0;
  register double b2 = 0;

  #pragma unroll 1
  for (int i = 0; i < k; i+=1){
    //load data
    a = *A;
    b1 = *B;
    b2 = *(B + ldb);
    A += lda;
    B += 1;

    //compute
    temp1 = temp1 + a * b1;
    temp2 = temp2 + a * b2;

  }

  *(C + 0 * ldc + idx) = temp1;
  *(C + 1 * ldc + idx) = temp2;
  
}


void test_kernel_naive(int m, int n, int k, 
            double * dA, int lda, 
            double * dB, int ldb, 
            double * dC, int ldc,
            float base){
  

for (int T = 16; T <= min(1024, m); T *= 2) {
   // int T = 128;
    int blocksPerGrid = m / T;
    int threadsPerBlock = T;


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TEST_RUN; i++)
      dgemm_kernel_naive<<<blocksPerGrid, threadsPerBlock>>>(m, n, k,
                  dA, lda, dB, ldb, dC, ldc);
      check_cuda_error();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float real_time = milliseconds / 1000;
    long long total_bytes = (m * k + k * n * (k / 32)) * sizeof(double);
    double total_gb = (double)total_bytes / 1e9;
    total_gb *= TEST_RUN;
    cout <<"Runing time of dgemm_kernel_naive("<< blocksPerGrid << "*" << T << "): " << real_time << " s" 
         <<" ("  << base/real_time <<"x)."
         <<" (" << total_gb <<"GB)"
         <<" (" << total_gb/real_time <<"GB/s)"<<endl;
  }

}


/////////////////////////SHARED/////////////////////////
__global__ void
dgemm_kernel_shared(int m, int n, int k, int T, double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  // store B (T * 2)
  extern __shared__ double cache[];
  
  //determine the row to process
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  register double temp1 = 0;
  register double temp2 = 0;
  register double a = 0;

  for (int j = 0; j < k; j += T){
    cache[threadIdx.x * 2] = *(B + threadIdx.x);
    cache[threadIdx.x * 2 + 1] = *(B + threadIdx.x + ldb);
    __syncthreads();
    B += T;
    for (int i = 0; i < T; i++) {
      a = *(A + (i + j) * lda);
      temp1 += a * cache[i * 2];
      temp2 += a * cache[i * 2 + 1];
    }
    __syncthreads();

  }
  *(C + 0 * ldc + idx) = temp1;
  *(C + 1 * ldc + idx) = temp2;

}


float test_kernel_shared(int m, int n, int k, 
          double * dA, int lda, 
          double * dB, int ldb, 
          double * dC, int ldc,
          float base){

    for (int T = 16; T <= min(1024, m); T *= 2) {

      //int T = 16;
      int blocksPerGrid = m / T;
      int threadsPerBlock = T;
      
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      for (int i = 0; i < TEST_RUN; i++)
        dgemm_kernel_shared<<<blocksPerGrid, threadsPerBlock,  T * sizeof(double) * 2>>>(m, n, k, T, dA, lda, dB, ldb, dC, ldc);
        check_cuda_error();
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      float real_time = milliseconds / 1000;
      long long total_bytes = (m * k + k * n * (k / T)) * sizeof(double) ;
      double total_gb = (double)total_bytes / 1e9;
      total_gb *= TEST_RUN;
      cout <<"Runing time of dgemm_kernel_shared("<< blocksPerGrid << "*" << T << "): " << real_time << "s" 
           <<" ("  << base/real_time <<"x)."
           <<" (" << total_gb <<"GB)"
           <<" (" << total_gb/real_time <<" GB/s)"<<endl;
    }
}

///////////////////////A PREFETCH(cache<->register)
__global__ void
dgemm_kernel_prefetch_s2r_16(int m, int n, int k, int T, double * A, int lda, double * B, int ldb, double * C, int ldc)
{

  extern __shared__ double cache[];
  
  double * cacheA = cache;
  double * cacheB = cache + T * T;
  
  //determine the row to process
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  double temp1 = 0;
  double temp2 = 0;

//prefectch A
  for (int i = 0; i < T; i++){
    cacheA[threadIdx.x + i * T] = *(A + i * lda);
  }
  
  double r0, r1, r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

  for (int j = 0; j < k; j += T){
    
    __syncthreads();
    cacheB[threadIdx.x * 2] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 2 + 1] = *(B + threadIdx.x + ldb);
    __syncthreads();
    B += T;

    if (j + T < k) {  
      A = A + T * lda;
      
      r0 = *(A + 0 *lda);
      r1 = *(A + 1 *lda);
      r2 = *(A + 2 *lda);
      r3 = *(A + 3 *lda);   
      r4 = *(A + 4 *lda);
      r5 = *(A+ 5 *lda);
      r6 = *(A + 6 *lda);
      r7 = *(A + 7 *lda);

      r8 = *(A + 8 *lda);
      r9 = *(A + 9 *lda);
      r10 = *(A + 10 *lda);
      r11 = *(A + 11 *lda);
      r12 = *(A + 12 *lda);
      r13 = *(A + 13 *lda);
      r14 = *(A + 14 *lda);
      r15 = *(A + 15 *lda);
    }

    for (int i = 0; i < T; i++) {      
      temp1 += cacheA[threadIdx.x +i * T] * cacheB[i * 2];
      temp2 += cacheA[threadIdx.x +i * T] * cacheB[i * 2 + 1];
    }
    if (j + T < k) {
      cacheA[threadIdx.x + 0 * T] = r0;
      cacheA[threadIdx.x + 1 * T] = r1;
      cacheA[threadIdx.x + 2 * T] = r2;
      cacheA[threadIdx.x + 3 * T] = r3;
      cacheA[threadIdx.x + 4 * T] = r4;
      cacheA[threadIdx.x + 5 * T] = r5;
      cacheA[threadIdx.x + 6 * T] = r6;
      cacheA[threadIdx.x + 7 * T] = r7;

      cacheA[threadIdx.x + 8 * T] = r8;
      cacheA[threadIdx.x + 9 * T] = r9;
      cacheA[threadIdx.x + 10 * T] = r10;
      cacheA[threadIdx.x + 11 * T] = r11;
      cacheA[threadIdx.x + 12 * T] = r12;
      cacheA[threadIdx.x + 13 * T] = r13;
      cacheA[threadIdx.x + 14 * T] = r14;
      cacheA[threadIdx.x + 15 * T] = r15;
    }

  }
  *(C + 0 * ldc + idx) = temp1;
  *(C + 1 * ldc + idx) = temp2;

}


__global__ void
dgemm_kernel_prefetch_s2r_8(int m, int n, int k, int T, double * A, int lda, double * B, int ldb, double * C, int ldc)
{

  extern __shared__ double cache[];
  
  double * cacheA = cache;
  double * cacheB = cache + T * T;
  
  //determine the row to process
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  double temp1 = 0;
  double temp2 = 0;

//prefectch A
  for (int i = 0; i < T; i++){
    cacheA[threadIdx.x + i * T] = *(A + i * lda);
  }
  
  double r0, r1, r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

  for (int j = 0; j < k; j += T){
    
    __syncthreads();
    cacheB[threadIdx.x * 2] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 2 + 1] = *(B + threadIdx.x + ldb);
    __syncthreads();
    B += T;

    if (j + T < k) {  
      A = A + T * lda;
      
      r0 = *(A + 0 *lda);
      r1 = *(A + 1 *lda);
      r2 = *(A + 2 *lda);
      r3 = *(A + 3 *lda);   
      r4 = *(A + 4 *lda);
      r5 = *(A+ 5 *lda);
      r6 = *(A + 6 *lda);
      r7 = *(A + 7 *lda);
    }

    for (int i = 0; i < T; i++) {      
      temp1 += cacheA[threadIdx.x +i * T] * cacheB[i * 2];
      temp2 += cacheA[threadIdx.x +i * T] * cacheB[i * 2 + 1];
    }
    if (j + T < k) {
      cacheA[threadIdx.x + 0 * T] = r0;
      cacheA[threadIdx.x + 1 * T] = r1;
      cacheA[threadIdx.x + 2 * T] = r2;
      cacheA[threadIdx.x + 3 * T] = r3;
      cacheA[threadIdx.x + 4 * T] = r4;
      cacheA[threadIdx.x + 5 * T] = r5;
      cacheA[threadIdx.x + 6 * T] = r6;
      cacheA[threadIdx.x + 7 * T] = r7;
    }

  }
  *(C + 0 * ldc + idx) = temp1;
  *(C + 1 * ldc + idx) = temp2;

}


void test_kernel_prefetch(int m, int n, int k, 
            double * dA, int lda, 
            double * dB, int ldb, 
            double * dC, int ldc,
            float base){

    for (int T = 8; T <= 16; T *= 2) {
    //int T = 16;
      int blocksPerGrid = m / T;
      int threadsPerBlock = T;

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      for (int i = 0; i < TEST_RUN; i++) {
        if (T == 16)
          dgemm_kernel_prefetch_s2r_16<<<blocksPerGrid, threadsPerBlock, ((T * 2) + (T * T)) * sizeof(double)>>>(m, n, k, T, dA, lda, dB, ldb, dC, ldc);
        else if (T == 8)
          dgemm_kernel_prefetch_s2r_8<<<blocksPerGrid, threadsPerBlock, ((T * 2) + (T * T)) * sizeof(double)>>>(m, n, k, T, dA, lda, dB, ldb, dC, ldc);
        check_cuda_error();
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      float real_time = milliseconds / 1000;
      long long total_bytes = (m * k + k * n * (k / T)) * sizeof(double) ;
        double total_gb = (double)total_bytes / 1e9;
        total_gb *= TEST_RUN;
        cout <<"Runing time of dgemm_kernel_prefetch("<< blocksPerGrid << "*" << T << "): " << real_time << "s" 
             <<" ("  << base/real_time <<"x)."
             <<" (" << total_gb <<"GB)"
             <<" (" << total_gb/real_time <<" GB/s)"<<endl;
    }
}



//Single registers: m, n, k, T, t, lda, ldb, ldc, idx, i, j, l (12)
//Double registers: cache, cacheA, cacheB, A, B, C, r0-3, temp1-2 (22)
//Shared mem.: T*2 + T*T (double)
__global__ void
dgemm_kernel_prefetch_s2r_4_16(int m, int n, int k, int T, int t, double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  // store B (T * 2)                                                                                                                                                                                                                                                                       
  extern __shared__ double cache[];
 
  double * cacheA = cache;
  double * cacheB = cache + T * t; //32 threads * 8 elements

  //determine the row to process                                                                                                                                                                                                                                                           
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  double temp1 = 0;
  double temp2 = 0;

  #pragma unroll 1
  //prefectch A 
  for (int i = 0; i < t; i++){
    cacheA[threadIdx.x + i * T] = *(A + i * lda);
  }
  A += t * lda;

  double r0, r1, r2, r3;

  #pragma unroll 1
  for (int j = 0; j < k; j += T){ 
    __syncthreads();
    cacheB[threadIdx.x * 2] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 2 + 1] = *(B + threadIdx.x + ldb);
    __syncthreads();
    B += T;

    #pragma unroll 1
    for (int l = j; l < j + T; l += t){
      if (l + t < k) {
        r0 = *(A + 0 *lda);
        r1 = *(A + 1 *lda);
        r2 = *(A + 2 *lda);
        r3 = *(A + 3 *lda); 
      }

      #pragma unroll 1
      for (int i = 0; i < t; i++) {
        temp1 += cacheA[threadIdx.x +i * T] * cacheB[l - j + i ];
        temp2 += cacheA[threadIdx.x +i * T] * cacheB[l - j + i + 1];
      }
      if (l + t < k) {
      cacheA[threadIdx.x + 0 * T] = r0;
      cacheA[threadIdx.x + 1 * T] = r1;
      cacheA[threadIdx.x + 2 * T] = r2;
      cacheA[threadIdx.x + 3 * T] = r3;
      }
      A += t * lda;
    }
  }
  *(C + 0 * ldc + idx) = temp1;
  *(C + 1 * ldc + idx) = temp2;
    
}

void test_kernel_prefetch2(int m, int n, int k, 
            double * dA, int lda, 
            double * dB, int ldb, 
            double * dC, int ldc,
            float base){    
    int T = 64;
    int tt = 4;
    int blocksPerGrid = m / T;
    int threadsPerBlock = T;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TEST_RUN; i++){
      dgemm_kernel_prefetch_s2r_4_16<<<blocksPerGrid, threadsPerBlock, ((T * 2) + (T * tt)) * sizeof(double)>>>(m, n, k, T, tt, dA, lda, dB, ldb, dC, ldc);
      check_cuda_error();
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float real_time = milliseconds / 1000;
    long long total_bytes = (m * k + k * n * (m / T)) * sizeof(double) ;
    double total_gb = (double)total_bytes / 1e9;
    total_gb *= TEST_RUN;
    cout <<"Runing time of dgemm_kernel_prefetch2("<< blocksPerGrid << "*" << T << "): " << real_time << "s" 
         <<" ("  << base/real_time <<"x)."
         <<" (" << total_gb <<"GB)"
         <<" (" << total_gb/real_time <<" GB/s)"<<endl;

}



//Single registers: m, n, k, T, t, lda, ldb, ldc, idx, j, l (11)
//Double registers: cacheB, A, B, C, nr0-3, cr0-3, temp1-2 (28)
//Shared mem.: T*2 + T*T (double)
//#define t 4
__global__ void
dgemm_kernel4_2(int m, int n, int k, int T, int t, double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  // store B (T * 2)                                                                                                                                                                                                                                                                       
  extern __shared__ double cacheB[];

  //determine the row to process                                                                                                                                                                                                                          
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  C = C + idx;
  register double temp1 = 0;
  register double temp2 = 0;

  register double nr0, nr1, nr2, nr3;
  register double cr0, cr1, cr2, cr3;

  //prefectch A 
  cr0 = *A;
  A += lda;
  cr1 = *A;
  A += lda;
  
  cr2 = *A;
  A += lda;
  cr3 = *A;
  A += lda;

  #pragma unroll 1
  for (int j = 0; j < k; j += T){ 
    __syncthreads();
    cacheB[threadIdx.x * 2] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 2 + 1] = *(B + threadIdx.x + ldb);
    __syncthreads();
    B += T;

    #pragma unroll 1
    for (int l = j; l < j + T; l += t){
      if (l + t < k) {
        nr0 = *A;
        A += lda;
        nr1 = *A;
        A += lda;

        nr2 = *A;
        A += lda;
        nr3 = *A;
        A += lda;
      }

      temp1 += cr0 * cacheB[l - j + 0 ];
      temp2 += cr0 * cacheB[l - j + 0 + 1];

      temp1 += cr1 * cacheB[l - j + 1 ];
      temp2 += cr1 * cacheB[l - j + 1 + 1];

     temp1 += cr2 * cacheB[l - j + 2 ];
     temp2 += cr2 * cacheB[l - j + 2 + 1];

     temp1 += cr3 * cacheB[l - j + 3 ];
     temp2 += cr3 * cacheB[l - j + 3 + 1];

      if (l + t < k) {
        cr0 = nr0;
        cr1 = nr1;
        cr2 = nr2;
        cr3 = nr3;
      }
    }
  }
  *C = temp1;
  *(C + ldc) = temp2;
    
}


//Single registers: m, n, k, T, t, lda, ldb, ldc, idx, j, l (11)
//Double registers: cacheB, A, B, C, nr0-3, cr0-3, temp1-2 (28)
//Shared mem.: T*2 + T*T (double)
__global__ void
dgemm_kernel4_3(int m, int n, int k, int T, int t, double * A, int lda, double * B, int ldb, double * C, int ldc)
{
                                                                                                                                                                                                                
  //determine the row to process                                                                                                                                                                                                                          
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  C = C + idx;
  register double temp1 = 0;
  register double temp2 = 0;

  register double nr0, nr1, nr2, nr3;
  register double cr0, cr1, cr2, cr3;

  register double nb00, nb01, nb10, nb11;
  register double cb00, cb01, cb10, cb11;

  //prefectch A 
  cr0 = *A;
  A += lda;
  cr1 = *A;
  A += lda;
  
  cr2 = *A;
  A += lda;
  cr3 = *A;
  A += lda;

  cb00 = *B;
  cb01 = *(B + ldb);
  B += 1;
  cb10 = *B;
  cb11 = *(B + ldb);
  B += 1;


  #pragma unroll 1
  for (int i = 0; i < k; i += t){ 
      if (i + t < k) {
        nr0 = *A;
        A += lda;
        nr1 = *A;
        A += lda;

        nr2 = *A;
        A += lda;
        nr3 = *A;
        A += lda;
      }
      
      //temp1 += cr1 * cr1;
      //temp2 += cr2 * cr2;
      //temp1 += cr3 * cr3;
      //temp1 += cr0 * cr0;
      
      nb00 = *B;
      nb01 = *(B + ldb);
      B += 1;
      nb10 = *B;
      nb11 = *(B + ldb);
      B += 1;

      temp1 += cr0 * cb00;
      temp2 += cr0 * cb01;
      temp1 += cr1 * cb10;
      temp2 += cr1 * cb11;

      cb00 = nb00;
      cb01 = nb01;
      cb10 = nb10;
      cb11 = nb11;


      if (i + t < k) {
        nb00 = *B;
        nb01 = *(B + ldb);
        B += 1;
        nb10 = *B;
        nb11 = *(B + ldb);
        B += 1;
      }

      temp1 += cr2 * cb00;
      temp2 += cr2 * cb01;
      temp1 += cr3 * cb10;
      temp2 += cr3 * cb11;

      cb00 = nb00;
      cb01 = nb01;
      cb10 = nb10;
      cb11 = nb11;
    
      
      if (i + t < k) {
        cr0 = nr0;
        cr1 = nr1;
        cr2 = nr2;
        cr3 = nr3;
      }
      
  }
  *C = temp1;
  *(C + ldc) = temp2;
    
}


float test_kernel_prefetch3(int m, int n, int k, 
            double * dA, int lda, 
            double * dB, int ldb, 
            double * dC, int ldc,
            float base){

    for (int T = 4; T <= min(m, 1024); T*=2) {
   
      int tt = 4;
      int blocksPerGrid = m / T;
      int threadsPerBlock = T;

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      for (int i = 0; i < TEST_RUN; i++) {
        dgemm_kernel4_2<<<blocksPerGrid, threadsPerBlock, ((T * 2)) * sizeof(double)>>>(m, n, k, T, tt, dA, lda, dB, ldb, dC, ldc);
        check_cuda_error();
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      float real_time = milliseconds / 1000;
      long long total_bytes = (m * k + k * n * (k / T)) * sizeof(double) ;
      double total_gb = (double)total_bytes / 1e9;
      total_gb *= TEST_RUN;
      cout <<"Runing time of dgemm_kernel_prefetch3("<< blocksPerGrid << "*" << T << "): " << real_time << "s" 
           <<" ("  << base/real_time <<"x)."
           <<" (" << total_gb <<"GB)"
           <<" (" << total_gb/real_time <<" GB/s)"<<endl;
    }

}


float test_kernel_prefetch4(int m, int n, int k, 
            double * dA, int lda, 
            double * dB, int ldb, 
            double * dC, int ldc,
            float base){

    for (int T = 16; T <= min(m, 1024); T*=2) {
    //int T = 128;
    int tt = 4;
      int blocksPerGrid = m / T;
      int threadsPerBlock = T;

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      for (int i = 0; i < TEST_RUN; i++) {
        dgemm_kernel4_3<<<blocksPerGrid, threadsPerBlock>>>(m, n, k, T, tt, dA, lda, dB, ldb, dC, ldc);
        check_cuda_error();
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      float real_time = milliseconds / 1000;
      long long total_bytes = (m * k + k * n * (k / 32)) * sizeof(double) ;
      double total_gb = (double)total_bytes / 1e9;
      total_gb *= TEST_RUN;
      cout <<"Runing time of dgemm_kernel_prefetch4("<< blocksPerGrid << "*" << T << "): " << real_time << "s" 
           <<" ("  << base/real_time <<"x)."
           <<" (" << total_gb <<"GB)"
           <<" (" << total_gb/real_time <<" GB/s)"<<endl;
    }

}




float test_cublas_mm(int m, int n, int k, 
            double * dA, int lda, 
            double * dB, int ldb, 
            double * dC, int ldc);


void test(int m, int k);

int main(){
  for (int i = 20480; i <= 30720; i += 1024){
  //int i = 1024;
    cout << "Test on: A (" << i << " x " << i << ") by B (" << i << " x " << 2 << ")" << endl;
    test(i, i);
  }
}

void test(int m, int k){
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    //int m = 20480;
    int n = 2;
    //int k = 20480;
    double * A = new double[m * k];
    double * B = new double[n * k];
    double * C = new double[m * n];
    double * checkC = new double[m * n];     

    for (int i = 0; i < m * k; i++){
    	A[i] = i;
    }

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
    
    float base;

    base = test_cublas_mm(m, n, k,  dA, lda, dB, ldb, dcheckC, ldc);
  
    test_kernel_naive(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    test_kernel_shared(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    test_kernel_prefetch(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    test_kernel_prefetch2(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    test_kernel_prefetch3(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    test_kernel_prefetch4(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    
   
    cudaMemcpy(C, dC ,m * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(checkC, dcheckC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    //for (int i = 0; i < m * n; i++){
    // cout<<C[i]<<" ";	
    //}
    check_C(C, m, n, checkC);

    //free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] checkC;

}



float test_cublas_mm(int m, int n, int k, 
         double * dA, int lda, 
         double * dB, int ldb, 
         double * dC, int ldc){

    double one = 1;
    double zero = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TEST_RUN; i++)
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        &one, dA, lda, dB, ldb, &zero, dC, ldc);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float real_time = milliseconds / 1000;

    cout <<"Runing time of culasdgemm:" << real_time <<" s." << endl;
    return real_time;
}






























