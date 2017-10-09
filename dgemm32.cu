#include <stdlib.h>
#include <iostream>
#include "cublas_v2.h"
#include <cmath>
#include <time.h>
#include <stdio.h>
#define TEST_RUN 1 
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
  //cout << "correct" << endl;
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
  register double temp3 = 0;
  register double temp4 = 0;
  register double a = 0;
  register double b1 = 0;
  register double b2 = 0;
  register double b3 = 0;
  register double b4 = 0;

  #pragma unroll 1
  for (int i = 0; i < k; i+=1){
    //load data
    a = *A;
    b1 = *B;
    b2 = *(B + ldb);
    b3 = *(B + ldb * 2);
    b4 = *(B + ldb * 3);
    A += lda;
    B += 1;

    //compute
    temp1 = temp1 + a * b1;
    temp2 = temp2 + a * b2;
    temp3 = temp3 + a * b3;
    temp4 = temp4 + a * b4;

  }

  *(C + 0 * ldc + idx) = temp1;
  *(C + 1 * ldc + idx) = temp2;
  *(C + 2 * ldc + idx) = temp3;
  *(C + 3 * ldc + idx) = temp4;
  
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
    long long total_bytes = (m * k + k * 4 * (k / 16)) * sizeof(double);
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
  register double temp3 = 0;
  register double temp4 = 0;
  register double a = 0;

  for (int j = 0; j < k; j += T){
    cache[threadIdx.x * 2] = *(B + threadIdx.x);
    cache[threadIdx.x * 2 + 1] = *(B + threadIdx.x + ldb);
    cache[threadIdx.x * 2 + 2] = *(B + threadIdx.x + ldb * 2);
    cache[threadIdx.x * 2 + 3] = *(B + threadIdx.x + ldb * 3);
    __syncthreads();
    B += T;
    for (int i = 0; i < T; i++) {
      a = *(A + (i + j) * lda);
      temp1 += a * cache[i * 2];
      temp2 += a * cache[i * 2 + 1];
      temp3 += a * cache[i * 2 + 2];
      temp4 += a * cache[i * 2 + 3];
    }
    __syncthreads();

  }
  *(C + 0 * ldc + idx) = temp1;
  *(C + 1 * ldc + idx) = temp2;
  *(C + 2 * ldc + idx) = temp3;
  *(C + 3 * ldc + idx) = temp4;

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
        dgemm_kernel_shared<<<blocksPerGrid, threadsPerBlock,  T * sizeof(double) * 4>>>(m, n, k, T, dA, lda, dB, ldb, dC, ldc);
        check_cuda_error();
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      float real_time = milliseconds / 1000;
      long long total_bytes = (m * k + k * 4 * (k / T)) * sizeof(double) ;
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
  double temp3 = 0;
  double temp4 = 0;

//prefectch A
  for (int i = 0; i < T; i++){
    cacheA[threadIdx.x + i * T] = *(A + i * lda);
  }
  
  double r0, r1, r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

  for (int j = 0; j < k; j += T){
    
    __syncthreads();
    cacheB[threadIdx.x * 4] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 4 + 1] = *(B + threadIdx.x + ldb);
    cacheB[threadIdx.x * 4 + 2] = *(B + threadIdx.x + ldb * 2);
    cacheB[threadIdx.x * 4 + 3] = *(B + threadIdx.x + ldb * 3);
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
      temp1 += cacheA[threadIdx.x +i * T] * cacheB[i * 4];
      temp2 += cacheA[threadIdx.x +i * T] * cacheB[i * 4 + 1];
      temp3 += cacheA[threadIdx.x +i * T] * cacheB[i * 4 + 2];
      temp4 += cacheA[threadIdx.x +i * T] * cacheB[i * 4 + 3];
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
  *(C + 2 * ldc + idx) = temp3;
  *(C + 3 * ldc + idx) = temp4;

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
  double temp3 = 0;
  double temp4 = 0;

//prefectch A
  for (int i = 0; i < T; i++){
    cacheA[threadIdx.x + i * T] = *(A + i * lda);
  }
  
  double r0, r1, r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

  for (int j = 0; j < k; j += T){
    
    __syncthreads();
    cacheB[threadIdx.x * 4] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 4 + 1] = *(B + threadIdx.x + ldb);
    cacheB[threadIdx.x * 4 + 2] = *(B + threadIdx.x + ldb * 2);
    cacheB[threadIdx.x * 4 + 3] = *(B + threadIdx.x + ldb * 3);
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
      temp1 += cacheA[threadIdx.x +i * T] * cacheB[i * 4];
      temp2 += cacheA[threadIdx.x +i * T] * cacheB[i * 4 + 1];
      temp3 += cacheA[threadIdx.x +i * T] * cacheB[i * 4 + 2];
      temp4 += cacheA[threadIdx.x +i * T] * cacheB[i * 4 + 3];
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
  *(C + 2 * ldc + idx) = temp3;
  *(C + 3 * ldc + idx) = temp4;

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
          dgemm_kernel_prefetch_s2r_16<<<blocksPerGrid, threadsPerBlock, ((T * 4) + (T * T)) * sizeof(double)>>>(m, n, k, T, dA, lda, dB, ldb, dC, ldc);
        else if (T == 8)
          dgemm_kernel_prefetch_s2r_8<<<blocksPerGrid, threadsPerBlock, ((T * 4) + (T * T)) * sizeof(double)>>>(m, n, k, T, dA, lda, dB, ldb, dC, ldc);
        check_cuda_error();
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      float real_time = milliseconds / 1000;
      long long total_bytes = (m * k + k * 4 * (k / T)) * sizeof(double) ;
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
  double * cacheB = cache + T * t; // 16 threads * 8 elements

  //determine the row to process                                                                                                                                                                                                                                                           
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  double temp1 = 0;
  double temp2 = 0;
  double temp3 = 0;
  double temp4 = 0;

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
    cacheB[threadIdx.x * 4] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 4 + 1] = *(B + threadIdx.x + ldb);
    cacheB[threadIdx.x * 4 + 2] = *(B + threadIdx.x + ldb * 2);
    cacheB[threadIdx.x * 4 + 3] = *(B + threadIdx.x + ldb * 3);
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
        temp3 += cacheA[threadIdx.x +i * T] * cacheB[l - j + i + 2];
        temp4 += cacheA[threadIdx.x +i * T] * cacheB[l - j + i + 3];
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
  *(C + 2 * ldc + idx) = temp3;
  *(C + 3 * ldc + idx) = temp4;
    
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
      dgemm_kernel_prefetch_s2r_4_16<<<blocksPerGrid, threadsPerBlock, ((T * 4) + (T * tt)) * sizeof(double)>>>(m, n, k, T, tt, dA, lda, dB, ldb, dC, ldc);
      check_cuda_error();
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float real_time = milliseconds / 1000;
    long long total_bytes = (m * k + k * 4 * (k / T)) * sizeof(double) ;
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
  register double temp3 = 0;
  register double temp4 = 0;
  register double temp5 = 0;
  register double temp6 = 0;
  register double temp7 = 0;
  register double temp8 = 0;
  register double temp9 = 0;
  register double temp10 = 0;
  register double temp11 = 0;
  register double temp12 = 0;
  register double temp13 = 0;
  register double temp14 = 0;
  register double temp15 = 0;
  register double temp16 = 0;
  register double temp17 = 0;
  register double temp18 = 0;
  register double temp19 = 0;
  register double temp20 = 0;
  register double temp21 = 0;
  register double temp22 = 0;
  register double temp23 = 0;
  register double temp24 = 0;
  register double temp25 = 0;
  register double temp26 = 0;
  register double temp27 = 0;
  register double temp28 = 0;
  register double temp29 = 0;
  register double temp30 = 0;
  register double temp31 = 0;
  register double temp32 = 0;


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
    cacheB[threadIdx.x * 16] = *(B + threadIdx.x);
    cacheB[threadIdx.x * 16 + 1] = *(B + threadIdx.x + ldb);
    cacheB[threadIdx.x * 16 + 2] = *(B + threadIdx.x + ldb * 2);
    cacheB[threadIdx.x * 16 + 3] = *(B + threadIdx.x + ldb * 3);
    cacheB[threadIdx.x * 16 + 4] = *(B + threadIdx.x + ldb * 4);
    cacheB[threadIdx.x * 16 + 5] = *(B + threadIdx.x + ldb * 5);
    cacheB[threadIdx.x * 16 + 6] = *(B + threadIdx.x + ldb * 6);
    cacheB[threadIdx.x * 16 + 7] = *(B + threadIdx.x + ldb * 7);
    cacheB[threadIdx.x * 16 + 8] = *(B + threadIdx.x + ldb * 8);
    cacheB[threadIdx.x * 16 + 9] = *(B + threadIdx.x + ldb * 9);
    cacheB[threadIdx.x * 16 + 10] = *(B + threadIdx.x + ldb * 10);
    cacheB[threadIdx.x * 16 + 11] = *(B + threadIdx.x + ldb * 11);
    cacheB[threadIdx.x * 16 + 12] = *(B + threadIdx.x + ldb * 12);
    cacheB[threadIdx.x * 16 + 13] = *(B + threadIdx.x + ldb * 13);
    cacheB[threadIdx.x * 16 + 14] = *(B + threadIdx.x + ldb * 14);
    cacheB[threadIdx.x * 16 + 15] = *(B + threadIdx.x + ldb * 15);
    cacheB[threadIdx.x * 16 + 16] = *(B + threadIdx.x + ldb * 16);
    cacheB[threadIdx.x * 16 + 17] = *(B + threadIdx.x + ldb * 17);
    cacheB[threadIdx.x * 16 + 18] = *(B + threadIdx.x + ldb * 18);
    cacheB[threadIdx.x * 16 + 19] = *(B + threadIdx.x + ldb * 19);
    cacheB[threadIdx.x * 16 + 20] = *(B + threadIdx.x + ldb * 20);
    cacheB[threadIdx.x * 16 + 21] = *(B + threadIdx.x + ldb * 21);
    cacheB[threadIdx.x * 16 + 22] = *(B + threadIdx.x + ldb * 22);
    cacheB[threadIdx.x * 16 + 23] = *(B + threadIdx.x + ldb * 23);
    cacheB[threadIdx.x * 16 + 24] = *(B + threadIdx.x + ldb * 24);
    cacheB[threadIdx.x * 16 + 25] = *(B + threadIdx.x + ldb * 25);
    cacheB[threadIdx.x * 16 + 26] = *(B + threadIdx.x + ldb * 26);
    cacheB[threadIdx.x * 16 + 27] = *(B + threadIdx.x + ldb * 27);
    cacheB[threadIdx.x * 16 + 28] = *(B + threadIdx.x + ldb * 28);
    cacheB[threadIdx.x * 16 + 29] = *(B + threadIdx.x + ldb * 29);
    cacheB[threadIdx.x * 16 + 30] = *(B + threadIdx.x + ldb * 30);
    cacheB[threadIdx.x * 16 + 31] = *(B + threadIdx.x + ldb * 31);
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
      temp3 += cr0 * cacheB[l - j + 0 + 2];
      temp4 += cr0 * cacheB[l - j + 0 + 3];
      temp5 += cr0 * cacheB[l - j + 0 + 4];
      temp6 += cr0 * cacheB[l - j + 0 + 5];
      temp7 += cr0 * cacheB[l - j + 0 + 6];
      temp8 += cr0 * cacheB[l - j + 0 + 7];
      temp9 += cr0 * cacheB[l - j + 0  + 8];
      temp10 += cr0 * cacheB[l - j + 0 + 9];
      temp11 += cr0 * cacheB[l - j + 0 + 10];
      temp12 += cr0 * cacheB[l - j + 0 + 11];
      temp13 += cr0 * cacheB[l - j + 0 + 12];
      temp14 += cr0 * cacheB[l - j + 0 + 13];
      temp15 += cr0 * cacheB[l - j + 0 + 14];
      temp16 += cr0 * cacheB[l - j + 0 + 15];
      temp17 += cr0 * cacheB[l - j + 0 + 16];
      temp18 += cr0 * cacheB[l - j + 0 + 17];
      temp19 += cr0 * cacheB[l - j + 0 + 18];
      temp20 += cr0 * cacheB[l - j + 0 + 19];
      temp21 += cr0 * cacheB[l - j + 0 + 20];
      temp22 += cr0 * cacheB[l - j + 0 + 21];
      temp23 += cr0 * cacheB[l - j + 0 + 22];
      temp24 += cr0 * cacheB[l - j + 0 + 23];
      temp25 += cr0 * cacheB[l - j + 0 + 24];
      temp26 += cr0 * cacheB[l - j + 0 + 25];
      temp27 += cr0 * cacheB[l - j + 0 + 26];
      temp28 += cr0 * cacheB[l - j + 0 + 27];
      temp29 += cr0 * cacheB[l - j + 0 + 28];
      temp30 += cr0 * cacheB[l - j + 0 + 29];
      temp31 += cr0 * cacheB[l - j + 0 + 30];
      temp32 += cr0 * cacheB[l - j + 0 + 31];

      temp1 += cr1 * cacheB[l - j + 1 ];
      temp2 += cr1 * cacheB[l - j + 1 + 1];
      temp3 += cr1 * cacheB[l - j + 1 + 2];
      temp4 += cr1 * cacheB[l - j + 1 + 3];
      temp5 += cr1 * cacheB[l - j + 1 + 4];
      temp6 += cr1 * cacheB[l - j + 1 + 5];
      temp7 += cr1 * cacheB[l - j + 1 + 6];
      temp8 += cr1 * cacheB[l - j + 1 + 7];
      temp9 += cr1 * cacheB[l - j + 1  + 8];
      temp10 += cr1 * cacheB[l - j + 1 + 9];
      temp11 += cr1 * cacheB[l - j + 1 + 10];
      temp12 += cr1 * cacheB[l - j + 1 + 11];
      temp13 += cr1 * cacheB[l - j + 1 + 12];
      temp14 += cr1 * cacheB[l - j + 1 + 13];
      temp15 += cr1 * cacheB[l - j + 1 + 14];
      temp16 += cr1 * cacheB[l - j + 1 + 15];
      temp17 += cr1 * cacheB[l - j + 1 + 16];
      temp18 += cr1 * cacheB[l - j + 1 + 17];
      temp19 += cr1 * cacheB[l - j + 1 + 18];
      temp20 += cr1 * cacheB[l - j + 1 + 19];
      temp21 += cr1 * cacheB[l - j + 1 + 20];
      temp22 += cr1 * cacheB[l - j + 1 + 21];
      temp23 += cr1 * cacheB[l - j + 1 + 22];
      temp24 += cr1 * cacheB[l - j + 1 + 23];
      temp25 += cr1 * cacheB[l - j + 1 + 24];
      temp26 += cr1 * cacheB[l - j + 1 + 25];
      temp27 += cr1 * cacheB[l - j + 1 + 26];
      temp28 += cr1 * cacheB[l - j + 1 + 27];
      temp29 += cr1 * cacheB[l - j + 1 + 28];
      temp30 += cr1 * cacheB[l - j + 1 + 29];
      temp31 += cr1 * cacheB[l - j + 1 + 30];
      temp32 += cr1 * cacheB[l - j + 1 + 31];

      temp1 += cr2 * cacheB[l - j + 2 ];
      temp2 += cr2 * cacheB[l - j + 2 + 1];
      temp3 += cr2 * cacheB[l - j + 2 + 2];
      temp4 += cr2 * cacheB[l - j + 2 + 3];
      temp5 += cr2 * cacheB[l - j + 2 + 4];
      temp6 += cr2 * cacheB[l - j + 2 + 5];
      temp7 += cr2 * cacheB[l - j + 2 + 6];
      temp8 += cr2 * cacheB[l - j + 2 + 7];
      temp9 += cr2 * cacheB[l - j + 2 + 8];
      temp10 += cr2 * cacheB[l - j + 2 + 9];
      temp11 += cr2 * cacheB[l - j + 2 + 10];
      temp12 += cr2 * cacheB[l - j + 2 + 11];
      temp13 += cr2 * cacheB[l - j + 2 + 12];
      temp14 += cr2 * cacheB[l - j + 2 + 13];
      temp15 += cr2 * cacheB[l - j + 2 + 14];
      temp16 += cr2 * cacheB[l - j + 2 + 15];
      temp17 += cr2 * cacheB[l - j + 2 + 16];
      temp18 += cr2 * cacheB[l - j + 2 + 17];
      temp19 += cr2 * cacheB[l - j + 2 + 18];
      temp20 += cr2 * cacheB[l - j + 2 + 19];
      temp21 += cr2 * cacheB[l - j + 2 + 20];
      temp22 += cr2 * cacheB[l - j + 2 + 21];
      temp23 += cr2 * cacheB[l - j + 2 + 22];
      temp24 += cr2 * cacheB[l - j + 2 + 23];
      temp25 += cr2 * cacheB[l - j + 2 + 24];
      temp26 += cr2 * cacheB[l - j + 2 + 25];
      temp27 += cr2 * cacheB[l - j + 2 + 26];
      temp28 += cr2 * cacheB[l - j + 2 + 27];
      temp29 += cr2 * cacheB[l - j + 2 + 28];
      temp30 += cr2 * cacheB[l - j + 2 + 29];
      temp31 += cr2 * cacheB[l - j + 2 + 30];
      temp32 += cr2 * cacheB[l - j + 2 + 31];


      temp1 += cr3 * cacheB[l - j + 3 ];
      temp2 += cr3 * cacheB[l - j + 3 + 1];
      temp3 += cr3 * cacheB[l - j + 3 + 2];
      temp4 += cr3 * cacheB[l - j + 3 + 3];
      temp5 += cr3 * cacheB[l - j + 3 + 4];
      temp6 += cr3 * cacheB[l - j + 3 + 5];
      temp7 += cr3 * cacheB[l - j + 3 + 6];
      temp8 += cr3 * cacheB[l - j + 3 + 7];
      temp9 += cr3 * cacheB[l - j + 3 + 8 ];
      temp10 += cr3 * cacheB[l - j + 3 + 9];
      temp11 += cr3 * cacheB[l - j + 3 + 10];
      temp12 += cr3 * cacheB[l - j + 3 + 11];
      temp13 += cr3 * cacheB[l - j + 3 + 12];
      temp14 += cr3 * cacheB[l - j + 3 + 13];
      temp15 += cr3 * cacheB[l - j + 3 + 14];
      temp16 += cr3 * cacheB[l - j + 3 + 15];
      temp17 += cr3 * cacheB[l - j + 3 + 16];
      temp18 += cr3 * cacheB[l - j + 3 + 17];
      temp19 += cr3 * cacheB[l - j + 3 + 18];
      temp20 += cr3 * cacheB[l - j + 3 + 19];
      temp21 += cr3 * cacheB[l - j + 3 + 20];
      temp22 += cr3 * cacheB[l - j + 3 + 21];
      temp23 += cr3 * cacheB[l - j + 3 + 22];
      temp24 += cr3 * cacheB[l - j + 3 + 23];
      temp25 += cr3 * cacheB[l - j + 3 + 24];
      temp26 += cr3 * cacheB[l - j + 3 + 25];
      temp27 += cr3 * cacheB[l - j + 3 + 26];
      temp28 += cr3 * cacheB[l - j + 3 + 27];
      temp29 += cr3 * cacheB[l - j + 3 + 28];
      temp30 += cr3 * cacheB[l - j + 3 + 29];
      temp31 += cr3 * cacheB[l - j + 3 + 30];
      temp32 += cr3 * cacheB[l - j + 3 + 31];

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
  *(C + ldc * 2) = temp3;
  *(C + ldc * 3) = temp4;
  *(C + ldc * 4) = temp5;
  *(C + ldc * 5) = temp6;
  *(C + ldc * 6) = temp7;
  *(C + ldc * 7) = temp8;
  *(C + ldc * 8) = temp9;
  *(C + ldc * 9) = temp10;
  *(C + ldc * 10) = temp11;
  *(C + ldc * 11) = temp12;
  *(C + ldc * 12) = temp13;
  *(C + ldc * 13) = temp14;
  *(C + ldc * 14) = temp15;
  *(C + ldc * 15) = temp16;
  *(C + ldc * 16) = temp17;
  *(C + ldc * 17) = temp18;
  *(C + ldc * 18) = temp19;
  *(C + ldc * 19) = temp20;
  *(C + ldc * 20) = temp21;
  *(C + ldc * 21) = temp22;
  *(C + ldc * 22) = temp23;
  *(C + ldc * 23) = temp24;
  *(C + ldc * 24) = temp25;
  *(C + ldc * 25) = temp26;
  *(C + ldc * 26) = temp27;
  *(C + ldc * 27) = temp28;
  *(C + ldc * 28) = temp29;
  *(C + ldc * 29) = temp30;
  *(C + ldc * 30) = temp31;
  *(C + ldc * 31) = temp32;

    
}


__global__ void
dgemm_kernel4_2_iter(int m, int n, int k, int T, int t, double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  // store B (T * 2)                                                                                                                                                                                                                                                                       
  extern __shared__ double cacheB[];

  //determine the row to process                                                                                                                                                                                                                          
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  A = A + idx;
  C = C + idx;
  B = B + threadIdx.x;
  register double temp1 = 0;
  register double temp2 = 0;
  register double temp3 = 0;
  register double temp4 = 0;
  register double temp5 = 0;
  register double temp6 = 0;
  register double temp7 = 0;
  register double temp8 = 0;
  register double temp9 = 0;
  register double temp10 = 0;
  register double temp11 = 0;
  register double temp12 = 0;
  register double temp13 = 0;
  register double temp14 = 0;
  register double temp15 = 0;
  register double temp16 = 0;



  register double nr0, nr1, nr2, nr3;
  register double cr0, cr1, cr2, cr3;

  //prefectch A 
  cr0 = *(A + lda * 0);
  cr1 = *(A + lda * 1);  
  cr2 = *(A + lda * 2);
  cr3 = *(A + lda * 3);



  #pragma unroll 1
  for (int j = 0; j < k; j += T){ 

    for (int p = 0; p < 2; p++){
      int b = p * 16;
      __syncthreads();
      for (int q = b; q < b + 16; q++){
        cacheB[threadIdx.x * 16 + q] = *(B + ldb * q);
      }
    // cacheB[threadIdx.x * 16 + 0] = *(B + ldb * 0);
    // cacheB[threadIdx.x * 16 + 1] = *(B + ldb * 1);
    // cacheB[threadIdx.x * 16 + 2] = *(B + ldb * 2);
    // cacheB[threadIdx.x * 16 + 3] = *(B + ldb * 3);
    // cacheB[threadIdx.x * 16 + 4] = *(B + ldb * 4);
    // cacheB[threadIdx.x * 16 + 5] = *(B + ldb * 5);
    // cacheB[threadIdx.x * 16 + 6] = *(B + ldb * 6);
    // cacheB[threadIdx.x * 16 + 7] = *(B + ldb * 7);
    // cacheB[threadIdx.x * 16 + 8] = *(B + ldb * 8);
    // cacheB[threadIdx.x * 16 + 9] = *(B + ldb * 9);
    // cacheB[threadIdx.x * 16 + 10] = *(B + ldb * 10);
    // cacheB[threadIdx.x * 16 + 11] = *(B + ldb * 11);
    // cacheB[threadIdx.x * 16 + 12] = *(B + ldb * 12);
    // cacheB[threadIdx.x * 16 + 13] = *(B + ldb * 13);
    // cacheB[threadIdx.x * 16 + 14] = *(B + ldb * 14);
    // cacheB[threadIdx.x * 16 + 15] = *(B + ldb * 15);
    __syncthreads();
  
    #pragma unroll 1
    for (int l = j; l < j + T; l += t){
      A = A + t * lda;

      if (p == 0) {
        if (l + t < j + T) {
          nr0 = *(A + lda * 0);
          nr1 = *(A + lda * 1);  
          nr2 = *(A + lda * 2);
          nr3 = *(A + lda * 3);

        } else {
          A = A - T * lda;
          nr0 = *(A + lda * 0);
          nr1 = *(A + lda * 1);  
          nr2 = *(A + lda * 2);
          nr3 = *(A + lda * 3);
        }
      } else {
        if (l + t < k) {
          nr0 = *(A + lda * 0);
          nr1 = *(A + lda * 1);  
          nr2 = *(A + lda * 2);
          nr3 = *(A + lda * 3);
        }
      }

      temp1 += cr0 * cacheB[l - j + 0 ];
      temp2 += cr0 * cacheB[l - j + 0 + 1];
      temp3 += cr0 * cacheB[l - j + 0 + 2];
      temp4 += cr0 * cacheB[l - j + 0 + 3];
      temp5 += cr0 * cacheB[l - j + 0 + 4];
      temp6 += cr0 * cacheB[l - j + 0 + 5];
      temp7 += cr0 * cacheB[l - j + 0 + 6];
      temp8 += cr0 * cacheB[l - j + 0 + 7];
      temp9 += cr0 * cacheB[l - j + 0  + 8];
      temp10 += cr0 * cacheB[l - j + 0 + 9];
      temp11 += cr0 * cacheB[l - j + 0 + 10];
      temp12 += cr0 * cacheB[l - j + 0 + 11];
      temp13 += cr0 * cacheB[l - j + 0 + 12];
      temp14 += cr0 * cacheB[l - j + 0 + 13];
      temp15 += cr0 * cacheB[l - j + 0 + 14];
      temp16 += cr0 * cacheB[l - j + 0 + 15];
      
      temp1 += cr1 * cacheB[l - j + 1 ];
      temp2 += cr1 * cacheB[l - j + 1 + 1];
      temp3 += cr1 * cacheB[l - j + 1 + 2];
      temp4 += cr1 * cacheB[l - j + 1 + 3];
      temp5 += cr1 * cacheB[l - j + 1 + 4];
      temp6 += cr1 * cacheB[l - j + 1 + 5];
      temp7 += cr1 * cacheB[l - j + 1 + 6];
      temp8 += cr1 * cacheB[l - j + 1 + 7];
      temp9 += cr1 * cacheB[l - j + 1  + 8];
      temp10 += cr1 * cacheB[l - j + 1 + 9];
      temp11 += cr1 * cacheB[l - j + 1 + 10];
      temp12 += cr1 * cacheB[l - j + 1 + 11];
      temp13 += cr1 * cacheB[l - j + 1 + 12];
      temp14 += cr1 * cacheB[l - j + 1 + 13];
      temp15 += cr1 * cacheB[l - j + 1 + 14];
      temp16 += cr1 * cacheB[l - j + 1 + 15];
      
      temp1 += cr2 * cacheB[l - j + 2 ];
      temp2 += cr2 * cacheB[l - j + 2 + 1];
      temp3 += cr2 * cacheB[l - j + 2 + 2];
      temp4 += cr2 * cacheB[l - j + 2 + 3];
      temp5 += cr2 * cacheB[l - j + 2 + 4];
      temp6 += cr2 * cacheB[l - j + 2 + 5];
      temp7 += cr2 * cacheB[l - j + 2 + 6];
      temp8 += cr2 * cacheB[l - j + 2 + 7];
      temp9 += cr2 * cacheB[l - j + 2 + 8];
      temp10 += cr2 * cacheB[l - j + 2 + 9];
      temp11 += cr2 * cacheB[l - j + 2 + 10];
      temp12 += cr2 * cacheB[l - j + 2 + 11];
      temp13 += cr2 * cacheB[l - j + 2 + 12];
      temp14 += cr2 * cacheB[l - j + 2 + 13];
      temp15 += cr2 * cacheB[l - j + 2 + 14];
      temp16 += cr2 * cacheB[l - j + 2 + 15];
     
      temp1 += cr3 * cacheB[l - j + 3 ];
      temp2 += cr3 * cacheB[l - j + 3 + 1];
      temp3 += cr3 * cacheB[l - j + 3 + 2];
      temp4 += cr3 * cacheB[l - j + 3 + 3];
      temp5 += cr3 * cacheB[l - j + 3 + 4];
      temp6 += cr3 * cacheB[l - j + 3 + 5];
      temp7 += cr3 * cacheB[l - j + 3 + 6];
      temp8 += cr3 * cacheB[l - j + 3 + 7];
      temp9 += cr3 * cacheB[l - j + 3 + 8 ];
      temp10 += cr3 * cacheB[l - j + 3 + 9];
      temp11 += cr3 * cacheB[l - j + 3 + 10];
      temp12 += cr3 * cacheB[l - j + 3 + 11];
      temp13 += cr3 * cacheB[l - j + 3 + 12];
      temp14 += cr3 * cacheB[l - j + 3 + 13];
      temp15 += cr3 * cacheB[l - j + 3 + 14];
      temp16 += cr3 * cacheB[l - j + 3 + 15];
      
      cr0 = nr0;
      cr1 = nr1;
      cr2 = nr2;
      cr3 = nr3;
    }

      *(C + ldc * (0 + b) )+= temp1;
      *(C + ldc * (1 + b) )+= temp2;
      *(C + ldc * (2 + b) )+= temp3;
      *(C + ldc * (3 + b) )+= temp4;
      *(C + ldc * (4 + b) )+= temp5;
      *(C + ldc * (5 + b) )+= temp6;
      *(C + ldc * (6 + b) )+= temp7;
      *(C + ldc * (7 + b) )+= temp8;
      *(C + ldc * (8 + b) )+= temp9;
      *(C + ldc * (9 + b) )+= temp10;
      *(C + ldc * (10 + b)) += temp11;
      *(C + ldc * (11 + b)) += temp12;
      *(C + ldc * (12 + b)) += temp13;
      *(C + ldc * (13 + b)) += temp14;
      *(C + ldc * (14 + b)) += temp15;
      *(C + ldc * (15 + b)) += temp16;


      temp1 = 0;
      temp2 = 0;
      temp3 = 0;
      temp4 = 0;
      temp5 = 0;
      temp6 = 0;
      temp7 = 0;
      temp8 = 0;
      temp9 = 0;
      temp10 = 0;
      temp11 = 0;
      temp12 = 0;
      temp13 = 0;
      temp14 = 0;
      temp15 = 0;
      temp16 = 0;

  }

  




  }



    
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
  register double temp3 = 0;
  register double temp4 = 0;

  register double nr0, nr1;//, nr2, nr3;
  register double cr0, cr1;//, cr2, cr3;

  register double nb00, nb01, nb02, nb03;//, nb10, nb11, nb12, nb13;
  register double cb00, cb01, cb02, cb03;//, cb10, cb11, cb12, cb13;

  //prefectch A 
  cr0 = *A;
  A += lda;
  cr1 = *A;
  A += lda;
  
  // cr2 = *A;
  // A += lda;
  // cr3 = *A;
  // A += lda;

  cb00 = *B;
  cb01 = *(B + ldb);
  cb02 = *(B + ldb * 2);
  cb03 = *(B + ldb * 3);
  B += 1;
  // cb10 = *B;
  // cb11 = *(B + ldb);
  // cb12 = *(B + ldb * 2);
  // cb13 = *(B + ldb * 3);
  // B += 1;


  #pragma unroll 1
  for (int i = 0; i < k; i += t){ 
      if (i + t < k) {
        nr0 = *A;
        A += lda;
        nr1 = *A;
        A += lda;

        // nr2 = *A;
        // A += lda;
        // nr3 = *A;
        // A += lda;
      }

      nb00 = *B;
      nb01 = *(B + ldb);
      // nb02 = *(B + ldb * 2);
      // nb03 = *(B + ldb * 3);
      B += 1;
      // nb10 = *B;
      // nb11 = *(B + ldb);
      // nb12 = *(B + ldb * 2);
      // nb13 = *(B + ldb * 3);
      // B += 1;

      temp1 += cr0 * cb00;
      temp2 += cr0 * cb01;
      temp3 += cr0 * cb02;
      temp4 += cr0 * cb03;

      // temp1 += cr1 * cb10;
      // temp2 += cr1 * cb11;
      // temp3 += cr1 * cb12;
      // temp4 += cr1 * cb13;

      cb00 = nb00;
      cb01 = nb01;
      cb02 = nb02;
      cb03 = nb03;

      // cb10 = nb10;
      // cb11 = nb11;
      // cb12 = nb12;
      // cb13 = nb13;

      nb00 = *B;
      nb01 = *(B + ldb);
      nb02 = *(B + ldb * 2);
      nb03 = *(B + ldb * 3);
      B += 1;

      temp1 += cr1 * cb00;
      temp2 += cr1 * cb01;
      temp3 += cr1 * cb02;
      temp4 += cr1 * cb03;

      cb00 = nb00;
      cb01 = nb01;
      cb02 = nb02;
      cb03 = nb03;



      // nb00 = *B;
      // nb01 = *(B + ldb);
      // nb02 = *(B + ldb * 2);
      // nb03 = *(B + ldb * 3);
      // B += 1;

      // temp1 += cr2 * cb00;
      // temp2 += cr2 * cb01;
      // temp3 += cr2 * cb02;
      // temp4 += cr2 * cb03;

      // cb00 = nb00;
      // cb01 = nb01;
      // cb02 = nb02;
      // cb03 = nb03;




      // if (i + t < k) {
      //   nb00 = *B;
      //   nb01 = *(B + ldb);
      //   nb02 = *(B + ldb * 2);
      //   nb03 = *(B + ldb * 3);
      //    B += 1;
      //   // nb10 = *B;
      //   // nb11 = *(B + ldb);
      //   // nb12 = *(B + ldb * 2);
      //   // nb13 = *(B + ldb * 3);
      //   // B += 1;
      // }

      // temp1 += cr3 * cb00;
      // temp2 += cr3 * cb01;
      // temp3 += cr3 * cb02;
      // temp4 += cr3 * cb03;
      // // temp1 += cr4 * cb10;
      // // temp2 += cr4 * cb11;

      // cb00 = nb00;
      // cb01 = nb01;
      // cb02 = nb02;
      // cb03 = nb03;

      // // cb10 = nb10;
      // // cb11 = nb11;
      // // cb12 = nb12;
      // // cb13 = nb13;
    

      if (i + t < k) {
        cr0 = nr0;
        cr1 = nr1;
        // cr2 = nr2;
        // cr3 = nr3;
      }
  }
  *C = temp1;
  *(C + ldc) = temp2;
  *(C + ldc * 2) = temp3;
  *(C + ldc * 3) = temp4;
    
}


float test_kernel_prefetch3(int m, int n, int k, 
            double * dA, int lda, 
            double * dB, int ldb, 
            double * dC, int ldc,
            float base){

    for (int T = 16; T <= min(m, 256); T*=2) {
    //int T = 128;
    int tt = 4;
      int blocksPerGrid = m / T;
      int threadsPerBlock = T;

      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaEventRecord(start);
      for (int i = 0; i < TEST_RUN; i++) {
        dgemm_kernel4_2_iter<<<blocksPerGrid, threadsPerBlock, (T * 16) * sizeof(double)>>>(m, n, k, T, tt, dA, lda, dB, ldb, dC, ldc);
        check_cuda_error();
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds, start, stop);

      float real_time = milliseconds / 1000;
      double read_a = ((m * k) / 1e9) * sizeof(double);
      double read_b = ((k * 16 * (k / T)) / 1e9) * sizeof(double);

      double total_gb = read_a + read_b;
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
    int tt = 2;
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
      long long total_bytes = (m * k + k * 4 * (k / T)) * sizeof(double) ;
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
  for (int i = 128; i <= 32768; i *= 2){
  //  int i = 6144;
    cout << "Test on: A (" << i << " x " << i << ") by B (" << i << " x " << 32 << ")" << endl;
    test(i, i);
  }
}

void test(int m, int k){
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    //int m = 20480;
    int n = 16;
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
    cudaMemcpy(checkC, dcheckC, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  
    // test_kernel_naive(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    // cudaMemcpy(C, dC ,m * n * sizeof(double), cudaMemcpyDeviceToHost);
    // check_C(C, m, n, checkC);


    // test_kernel_shared(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    // cudaMemcpy(C, dC ,m * n * sizeof(double), cudaMemcpyDeviceToHost);
    // check_C(C, m, n, checkC);

    // test_kernel_prefetch(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    // cudaMemcpy(C, dC ,m * n * sizeof(double), cudaMemcpyDeviceToHost);
    // check_C(C, m, n, checkC);

    // test_kernel_prefetch2(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    // cudaMemcpy(C, dC ,m * n * sizeof(double), cudaMemcpyDeviceToHost);
    // check_C(C, m, n, checkC);

    test_kernel_prefetch3(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    cudaMemcpy(C, dC ,m * n * sizeof(double), cudaMemcpyDeviceToHost);
    check_C(C, m, n, checkC);

    //test_kernel_prefetch4(m, n, k, dA, lda, dB, ldb, dC, ldc, base);
    
   
    //cudaMemcpy(C, dC ,m * n * sizeof(double), cudaMemcpyDeviceToHost);
    
    //for (int i = 0; i < m * n; i++){
    // cout<<C[i]<<" ";	
    //}
   // check_C(C, m, n, checkC);

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






























