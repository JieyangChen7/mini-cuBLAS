#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void array_generator(int n, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //clock_t start = clock();
  for (int i =0; i < 10; i++) {
    A[idx + blockDim.x * i] = (unsigned long long int)(A + idx + blockDim.x * (i + 1));
  }
//clock_t end = clock();
  //printf("%d\n", end-start);
}

__global__ void global_memory(int n, double * A, int space, int iteration, unsigned long long int * T) {
  int idx = blockIdx.x * space + threadIdx.x;
  A = A + idx;
  volatile clock_t start = 0;
  volatile clock_t end = 0;
  volatile unsigned long long sum_time = 0;
  register double a;
  
  for (int i = 0; i < iteration; i++) {
    start = clock();
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    A = (double *)(unsigned long long int) *A;
    end = clock();
    /*asm volatile ("{\n\t"
                  "mov.u32 %0, %%clock;\n\t"
                  "ld.global.f64 %1, [%3];\n\t"
		  "ld.global.f64 %3, [%1];\n\t"
                  "ld.global.f64 %1, [%3];\n\t"
		  "ld.global.f64 %3, [%1];\n\t"
		  "ld.global.f64 %1, [%3];\n\t"
                  "ld.global.f64 %3, [%1];\n\t"
		  "ld.global.f64 %1, [%3];\n\t"
                  "ld.global.f64 %3, [%1];\n\t"
		  "ld.global.f64 %1, [%3];\n\t"
                  "ld.global.f64 %3, [%1];\n\t"
		  "mov.u32 %2, %%clock;\n\t"
                  "}"
                  :  "=r"(start), "+d"(a), "=r"(end), "+d"(A):: "memory"
                  );
    */
    sum_time += (end - start);
  }
  T[idx] = sum_time;

  //printf("%d ", end-start);
  //printf("SE: %d %d", start, end);

}

__global__ void register_latency(int iteration, unsigned long long int * T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
   register int start = 0;
    register int end = 0;
   unsigned long long sum_time = 0;

   register int a = 1;
   register int b = 2;
   register int c = 3;
   for (int i = 0; i < iteration; i++) {
     asm volatile ("{\n\t"
                  "mov.u32 %0, %%clock;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mad.lo.s32 %1, %3, %4, %1;\n\t"
                  "mov.u32 %2, %%clock;\n\t"
                  "}"
                  :  "=r"(start), "=r"(a), "=r"(end): "r"(b), "r"(c) : "memory"
                  );

    sum_time += (end - start);
  }
  T[idx] = sum_time;
}

__global__ void global_memory_access_latency(int iteration, unsigned long long int * T, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  register int start = 0;
  register int end = 0;
  unsigned long long sum_time = 0;

  register double a = 1;
  
  for (int i = 0; i < iteration; i++) {
    asm volatile ("{\n\t"
                  "mov.u32 %0, %%clock;\n\t"
                  "ld.global.f64 %1, [%3];\n\t"
		  
                  "mov.u32 %2, %%clock;\n\t"
                  "}"
		  :  "=r"(start), "=d"(a), "=r"(end): "l"(A): "memory"
                  );

    sum_time += (end - start);
  }
  T[idx] = sum_time;
  A[idx] = a;

}


__global__ void ILP_latency(int iteration, unsigned long long int * T, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  register int start = 0;
  register int end = 0;
  unsigned long long sum_time = 0;

  register int a1 = 1;
  register int a2 = 2;
  register int a3 = 3;
  register int a4 = 4;
  register int a5 = 5;
  register int a6 = 6;
  register int a7 = 7;
  register int a8 = 8;
  register int a9 = 9;
  register int a10 = 10;
  for (int i = 0; i < iteration; i++) {
      asm volatile ("{\n\t"
                  "mov.u32 %0, %%clock;\n\t"
                  "mad.lo.s32 %1, %1, %1, %1;\n\t"
                  "mad.lo.s32 %2, %2, %2, %2;\n\t"
		  "mad.lo.s32 %3, %3, %3, %3;\n\t"
		  "mad.lo.s32 %4, %4, %4, %4;\n\t"
		  "mad.lo.s32 %5, %5, %5, %5;\n\t"
		  "mad.lo.s32 %6, %6, %6, %6;\n\t"
                  "mad.lo.s32 %7, %7, %7, %7;\n\t"
		  "mad.lo.s32 %8, %8, %8, %8;\n\t"
		  "mad.lo.s32 %9, %9, %9, %9;\n\t"
		  "mad.lo.s32 %10, %10, %10, %10;\n\t"
		  "mov.u32 %11, %%clock;\n\t"
                  "}"
		    :  "=r"(start), "=r"(a1),"=r"(a2), "=r"(a3), "=r"(a4), "=r"(a5),
		       "=r"(a6), "=r"(a7),"=r"(a8),"=r"(a9),"=r"(a10),"=r"(end):: "memory"
                  );

    sum_time += (end - start);
  }
  T[idx] = sum_time;
  A[idx] = a1 + a2 + a3 +a4 + a5 + a6 + a7 + a8 + a9 + a10;
  
}


__global__ void ILP_latency2(int iteration, unsigned long long int * T, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  register int start = 0;
  register int end = 0;
  unsigned long long sum_time = 0;

  register int a1 = 1;
  register int a2 = 2;
  register int a3 = 3;
  register int a4 = 4;
  register int a5 = 5;
  register int a6 = 6;
  register int a7 = 7;
  register int a8 = 8;
  register int a9 = 9;
  register int a10 = 10;
  for (int i = 0; i < iteration; i++) {
    asm volatile ("{\n\t"
                  "mov.u32 %0, %%clock;\n\t"
                  "add.s32 %1, %1, %1;\n\t"
                  "add.s32 %2, %2, %2;\n\t"
		  "add.s32 %3, %3, %3;\n\t"
		  "add.s32 %4, %4, %4;\n\t"
		  "add.s32 %5, %5, %5;\n\t"
		  "add.s32 %6, %6, %6;\n\t"
                  "add.s32 %7, %7, %7;\n\t"
                  "add.s32 %8, %8, %8;\n\t"
                  "add.s32 %9, %9, %9;\n\t"
                  "add.s32 %10, %10, %10;\n\t"
		  "mov.u32 %11, %%clock;\n\t"
                  "}"
		  :  "=r"(start), "=r"(a1),"=r"(a2),"=r"(a3), "=r"(a4),"=r"(a5),
		     "=r"(a6),"=r"(a7),"=r"(a8), "=r"(a9),"=r"(a10), "=r"(end) :: "memory"
                  );

    sum_time += (end - start);
  }
  T[idx] = sum_time;
  A[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10;

}



int main(){
  int n = 128;
  int B = 16;
  double * A = new double[n + B*10];
  unsigned long long int * T = new unsigned long long int[n];
  
  double * dA;
  unsigned long long int *dT;
  cudaMalloc(&dA, (n + B*10) * sizeof(double));
  cudaMalloc((void**)&dT, n * sizeof(unsigned long long int));

  array_generator<<<n/B, B>>>(n, dA);
  global_memory<<<n/B, B>>>(n, dA, B, 1, dT);
  //register_latency<<<n/B, B>>>(1, dT);
  //ILP_latency<<<n/B, B>>>(1, dT, dA);
  //global_memory_access_latency<<<n/B, B>>>(1, dT, dA); 
  cudaMemcpy(A, dA, (n + B*10) * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(T, dT, n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
 
//  for (int i = 0; i < n + B; i++) {
//  	cout << A[i] << " ";
//  }
  long long int sum = 0;
    for (int i = 0; i < n; i++) {
      sum += T[i];
      cout << "Thread.id[" << i << "]: latency "<< T[i] << " cycles"<< endl;;
    }
    cout << "Average latency is: " << (double)sum / n << "cycles" << endl;
}
