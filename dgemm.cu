#include <stdlib.h>
#include <iostream>
#include "cublas_v2.h"
#include "papi.h"
using namespace std;

#define NB 8
#define T 2 // tile

__global__ void
dgemm_kernel1(double * A,int lda, double * B, int ldb, double * C, int ldc);
__global__ void
dgemm_kernel1_5(double * A,int lda, double * B, int ldb, double * C, int ldc);
__global__ void
dgemm_kernel2(double * A, int lda, double * B, int ldb, double * C, int ldc);
__global__ void
dgemm_kernel3(double * A, int lda, double * B, int ldb, double * C, int ldc);
__global__ void
dgemm_kernel3_P(double * A, int lda, double * B, int ldb, double * C, int ldc);


int main(){
    int m = 8;
    int n = 8;
    double * A = new double[n * m];
    double * B = new double[m];
    double * C = new double[n];    

    for (int i=0;i<n*m;i++){
    	A[i] = i;
    }

    //for (int i = 0; i < m; i ++){
    // 	for (int j = 0; j < n; j++) {
    //	    cout << A[i + j * 8] << " ";
    //	}
    //	cout << endl;
    //}      
    
    for (int i=0;i<m;i++){
    	B[i] = 1;
    }
    
    double * dA;
    cudaMalloc(&dA, n * m * sizeof(double));
    
    double * dB;
    cudaMalloc(&dB,  m * sizeof(double));

    double * dC;
    cudaMalloc(&dC, n * sizeof(double));

    cudaMemcpy(dA,A,n * m * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,m * sizeof(double),cudaMemcpyHostToDevice);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    float real_time = 0.0;
    float proc_time = 0.0;
    long long flpins = 0.0;
    float mflops = 0.0;

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
      //return -1;
    }

    double one = 1;
    double zero = 0;
    cublasDgemv(handle,CUBLAS_OP_N, 8, 8, &one, dA, 8, dB, 1, &zero,dC, 1);
    cudaDeviceSynchronize();

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
	cout << "PAPI ERROR" << endl;
	//return -1;                                                                                    
      }
    
    cout <<"Runing time of culasdgemv:" << real_time <<"s." << endl;
    

    //set timer to zero
    real_time = 0.0;
    proc_time = 0.0;
    flpins = 0.0;
    mflops = 0.0;

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
      //return -1;
    }



    int threadsPerBlock = 8;
    int blocksPerGrid = 8;
    dgemm_kernel1<<<blocksPerGrid, threadsPerBlock>>>(dA, 8, dB, 8, dC, 8);

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
      //return -1;
    }
    
    cout <<"Runing time of dgemm_kernel1: " << real_time << "s." << endl;
    
    real_time = 0.0;
    proc_time = 0.0;
    flpins = 0.0;
    mflops = 0.0;

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
	//return -1;
    }

    dgemm_kernel1_5<<<blocksPerGrid, threadsPerBlock>>>(dA, 8, dB, 8, dC, 8);

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
      //return -1;
    }

    cout <<"Runing time of dgemm_kernel1_5: " << real_time << "s." << endl;


    real_time = 0.0;
    proc_time = 0.0;
    flpins = 0.0;
    mflops = 0.0;

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
        //return -1
    }

    dgemm_kernel2<<<4,2>>>(dA, 8, dB, 8, dC, 8);

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
      //return -1;
    }

    cout <<"Runing time of dgemm_kernel2: " << real_time << "s." << endl;    


    real_time = 0.0;
    proc_time = 0.0;
    flpins = 0.0;
    mflops = 0.0;

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
        //return -1;
    }

    dgemm_kernel3<<<4,2>>>(dA, 8, dB, 8, dC, 8);

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
      //return -1;
    }

    cout <<"Runing time of dgemm_kernel3: " << real_time << "s." << endl;


    real_time = 0.0;
    proc_time = 0.0;
    flpins = 0.0;
    mflops = 0.0;

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
        //return -1;
    }

    dgemm_kernel3_P<<<4,2>>>(dA, 8, dB, 8, dC, 8);

    if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
      cout << "PAPI ERROR" << endl;
      //return -1;
    }

    cout <<"Runing time of dgemm_kernel3_P: " << real_time << "s." << endl;


    //invoke kernel
    //int threadsPerBlock = 8;
    //int blocksPerGrid = 8;
    //dgemm_kernel1<<<blocksPerGrid, threadsPerBlock>>>(dA, 8, dB, 8, dC, 8);
    //dgemm_kernel1_5<<<blocksPerGrid, threadsPerBlock>>>(dA, 8, dB, 8, dC, 8);
    //dgemm_kernel2<<< 4, 2>>>(dA, 8, dB, 8, dC, 8);
    //dgemm_kernel3<<< 4, 2>>>(dA, 8, dB, 8, dC, 8);
    //dgemm_kernel3_P<<< 4, 2>>>(dA, 8, dB, 8, dC, 8); 
    // copy vectors from host memory to device memory
    cudaMemcpy(C,dC,n *sizeof(double),cudaMemcpyDeviceToHost);
    
    //    for (int i=0;i<n;i++){
    //	cout<<C[i]<<" ";	
    //}
    
    //free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    

}



__global__ void
dgemm_kernel1(double * A, int lda, double * B, int ldb, double * C , int ldc)  
{

    //blockIdx.x: determin the row to process
    A = A + blockIdx.x;
    B = B;

    __shared__ double cache[NB];

    //load one row to cache
    double a = *(A + lda * threadIdx.x);
    double b = *(B + threadIdx.x);
    cache[threadIdx.x] = a * b;

    __syncthreads();

	/* logrithm reduction */
    int i = blockDim.x / 2;
    while (i != 0) {
    if (threadIdx.x < i)
        cache[threadIdx.x] += cache[threadIdx.x + i];
	__syncthreads();
	i /= 2;
    }

    if (threadIdx.x == 0) {
	C[blockIdx.x] = cache[0];
    }
	
}


__global__ void
dgemm_kernel1_5(double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  //blockIdx.x: determin the row to process
  A = A + blockIdx.x ;
  B = B ;
  
  __shared__ double cache[NB];
  
  //load one column to cache
  double a = *(A + lda * threadIdx.x);
  double b = *(B + threadIdx.x);
  cache[threadIdx.x] = a * b;
  
  __syncthreads();

  double sum = 0;
  if (threadIdx.x == 0) {
    
    for (int i = 0; i < NB; i++) {
      sum += cache[i];
    }
    C[blockIdx.x] = sum;
  }
  
}

__global__ void
dgemm_kernel2(double * A, int lda, double * B, int ldb, double * C, int ldc)
{
	//determine the row to process
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	A = A + idx;

	double temp = 0;
	for (int i = 0;i < NB; i++){
	  double a = *(A + i * lda);
	  double b = *(B + i);
		temp = temp + a * b;
	}

	*(C + idx) = temp;

}


__global__ void
dgemm_kernel3(double * A, int lda, double * B, int ldb, double * C, int ldc)
{
  A = A + blockIdx.x * T;
  
  __shared__ double cache[T][T];
  
  double sum = 0;
  
  for (int i = 0; i < NB; i += T){
    //load a block to cache
    for (int j = 0; j < T; j++){
      cache[j][threadIdx.x] = *(A + (threadIdx.x + i) * lda + j);
    }
    __syncthreads();
    
    for (int j = 0; j < T; j++){
      sum = cache[threadIdx.x][j] * B[i + j] + sum;
    }

    __syncthreads();
    
  }
  
  *(C + blockIdx.x * T + threadIdx.x) = sum;
	
}



__global__ void
dgemm_kernel3_P(double * A, int lda, double * B, int ldb, double * C, int ldc)
{
	A = A + blockIdx.x * T;

	__shared__ double cache[T][T];

	double sum = 0;

	double r0 = *(A + threadIdx.x * lda + 0);
	double r1 = *(A + threadIdx.x * lda + 1);
	/*double r2 = *(A + threadIdx.x * lda + 2);
	double r3 = *(A + threadIdx.x * lda + 3);
	double r4 = *(A + threadIdx.x * lda + 4);
	double r5 = *(A + threadIdx.x * lda + 5);
	double r6 = *(A + threadIdx.x * lda + 6);
	double r7 = *(A + threadIdx.x * lda + 7);

	double r8 = *(A + threadIdx.x * lda + 8);
	double r9 = *(A + threadIdx.x * lda + 9);
	double r10 = *(A + threadIdx.x * lda + 10);
	double r11 = *(A + threadIdx.x * lda + 11);
	double r12 = *(A + threadIdx.x * lda + 12);
	double r13 = *(A + threadIdx.x * lda + 13);
	double r14 = *(A + threadIdx.x * lda + 14);
	double r15 = *(A + threadIdx.x * lda + 15);
	*/
	for (int i = 0; i < NB; i += T){

		//load current register to shared mem
		cache[0][threadIdx.x] = r0;
		cache[1][threadIdx.x] = r1;
		/*cache[2][threadIdx.x] = r2;
		cache[3][threadIdx.x] = r3;
		cache[4][threadIdx.x] = r4;
		cache[5][threadIdx.x] = r5;
		cache[6][threadIdx.x] = r6;
		cache[7][threadIdx.x] = r7;

		cache[8][threadIdx.x] = r8;
		cache[9][threadIdx.x] = r9;
		cache[10][threadIdx.x] = r10;
		cache[11][threadIdx.x] = r11;
		cache[12][threadIdx.x] = r12;
		cache[13][threadIdx.x] = r13;
		cache[14][threadIdx.x] = r14;
		cache[15][threadIdx.x] = r15;
		*/
		__syncthreads();

		A = A + T * lda;

		// load next block to register

		r0 = *(A + threadIdx.x * lda + 0);
	        r1 = *(A + threadIdx.x * lda + 1);
	 	/*r2 = *(A + threadIdx.x * lda + 2);
	 	r3 = *(A + threadIdx.x * lda + 3);
	 	r4 = *(A + threadIdx.x * lda + 4);
	 	r5 = *(A + threadIdx.x * lda + 5);
	 	r6 = *(A + threadIdx.x * lda + 6);
		r7 = *(A + threadIdx.x * lda + 7);

	 	r8 = *(A + threadIdx.x * lda + 8);
		r9 = *(A + threadIdx.x * lda + 9);
	 	r10 = *(A + threadIdx.x * lda + 10);
	 	r11 = *(A + threadIdx.x * lda + 11);
	 	r12 = *(A + threadIdx.x * lda + 12);
	 	r13 = *(A + threadIdx.x * lda + 13);
	 	r14 = *(A + threadIdx.x * lda + 14);
	 	r15 = *(A + threadIdx.x * lda + 15);	
		*/
		for (int j = 0; j < T; j++){
		  sum = cache[threadIdx.x][j] * B[i + j] + sum;
		}


	}

	*(C + blockIdx.x * T + threadIdx.x) = sum;

}




