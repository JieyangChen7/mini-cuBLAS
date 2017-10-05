#include <iostream>
#include <stdio.h>
#include <climits>
#include <algorithm>
using namespace std;

__global__ void array_generator(double * A, int iteration, int access_per_iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  idx = idx * access_per_iter;
  A = A + idx;
  for (int i = 0; i < iteration; i++) {
    for (int j = 0 ; j < access_per_iter; j++) {
      A[j] = (unsigned long long int)(A + j + gridDim.x * blockDim.x * access_per_iter );
      //if (idx == 0) {
      //	printf("%d-%d-%d\n",i,j, (unsigned long long int)(A + j + gridDim.x * blockDim.x * access_per_iter ));
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



__global__ void gl2r_latency(int iteration, unsigned long long int * dStart, unsigned long long int * dEnd,double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ double cache[];
   
  //register int start = 0;
  //register int end = 0;
  //unsigned long long sum_time = 0;
  A = A + idx * 32;
  register double * A1 = A;
  register double * A2 = A+1;
  register double * A3 = A+2;
  register double * A4 = A+3;
  register double * A5 = A+4;
  register double * A6 = A+5;
  register double * A7 = A+6;
  register double * A8 = A+7;
  register double * A9 = A+8;
  register double * A10 = A+9;
  register double * A11 = A+10;
  register double * A12 = A+11;
  register double * A13 = A+12;
  register double * A14 = A+13;
  register double * A15 = A+14;
  register double * A16 = A+15;
  register double * A17 = A+16;
  register double * A18 = A+17;
  register double * A19 = A+18;
  register double * A20 = A+19;
  register double * A21 = A+20;
  register double * A22 = A+21;
  register double * A23 = A+22;
  register double * A24 = A+23;
  register double * A25 = A+24;
  register double * A26 = A+25;
  register double * A27 = A+26;
  register double * A28 = A+27;
  register double * A29 = A+28;
  
  //register double * A30 = A+29;
  //register double * A31 = A+30;
  //register double * A32 = A+31;

  
  register double a1;
  register double a2;
  register double a3;
  register double a4;
  register double a5;
  register double a6;
  register double a7;
  register double a8;
  register double a9;
  register double a10;
  register double a11;
  register double a12;
  register double a13;
  register double a14;
  register double a15;
  register double a16;
  register double a17;
  register double a18;
  register double a19;
  register double a20;
  register double a21;
  register double a22;
  register double a23;
  register double a24;
  register double a25;
  register double a26;
  register double a27;
  register double a28;
  register double a29;
  
  //register double a30;
  //register double a31;
  //register double a32;

  for (int i = 0; i < iteration; i++) {
  //start = clock();
    a1 = *A1;
    a2 = *A2;
    a3 = *A3;
    a4 = *A4;
    a5 = *A5;
    a6 = *A6;
    a7 = *A7;
    a8 = *A8;
    a9 = *A9;
    a10 = *A10;
    a11 = *A11;
    a12 = *A12;
    a13 = *A13;
    a14 = *A14;
    a15 = *A15;
    a16 = *A16;
    
    a17 = *A17;
    a18 = *A18;
    a19 = *A19;
    a20 = *A20;
    a21 = *A21;
    a22 = *A22;
    a23 = *A23;
    a24 = *A24;
    a25 = *A25;
    a26 = *A26;
    a27 = *A27;
    a28 = *A28;
    a29 = *A29;
    
    //a30 = *A30;
    //a31 = *A31;
    //a32 = *A32;
    __syncthreads();
    a1 += a2;
    a3 += a4;
    a5 += a6;
    a7 += a8;
    a9 += a10;
    a11 += a12;
    a13 += a14;
    a15 += a16;
    
    a17 += a18;
    a19 += a20;
    a21 += a22;
    a23 += a24;
    a25 += a26;
    a27 += a28;
    
    //a29 += a30;
    //a31 += a32;
    
    
    a1 += a3;
    a5 += a7;
    a9 += a11;
    a13 += a15;
    
    a17 += a19;
    a21 += a23;
    a25 += a27;
    
    //a29 += a31;


    a1 += a5;
    a9 += a13;
    
    a17 += a21;
    a25 += a29;
    

    a1 += a9;
    
    a17 +=a25;
    
    a1 += a17;
    
    
 
    //end = clock();
    //sum_time += (end - start);
    }
  
 

    //dStart[idx] = start;
    //dEnd[idx] = end;
  
  A[idx] = a1;

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



__global__ void gl2s_latency(int iteration, unsigned long long int * T, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  register int start = 0;
  register int end = 0;
  unsigned long long sum_time = 0;

  register double * A1 = A;
  register double * A2 = A+1;
  register double * A3 = A+2;
  register double * A4 = A+3;
  register double * A5 = A+4;
  register double * A6 = A+5;
  register double * A7 = A+6;
  register double * A8 = A+7;
  __shared__ double a[8*32];
  double * s = a + idx * 8;
  for (int i = 0; i < iteration; i++) {
    start = clock();
    s[0] = *A1;
    s[1] = *A2;
    s[2] = *A3;
    s[3] = *A4;
    s[4] = *A5;
    s[5] = *A6;
    s[6] = *A7;
    s[7] = *A8;
    //a1 = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    __syncthreads();                                                                                                                      
    end = clock();
    /*    asm volatile ("{\n\t"                                                                                                             
                  "mov.u32 %0, %%clock;\n\t"                                                                                                
                  "ld.global.f64 %2, [%6];\n\t"                                                                                             
                  "ld.global.f64 %3, [%7];\n\t"                                                                                             
                  "ld.global.f64 %4, [%8];\n\t"                                                                                             
                  "ld.global.f64 %5, [%9];\n\t"                                                                                             
                  "bar.sync 1;\n\t"                                                                                                         
                  "mov.u32 %1, %%clock;\n\t"                                                                                                
                  "}"                                                                                                                       
                  :  "=r"(start), "=r"(end), "=d"(a1), "=d"(a2), "=d"(a3), "=d"(a4):                                                        
                     "l"(A1), "l"(A2), "l"(A3), "l"(A4): "memory"                                                                           
                  );                                                                                                                        
    */
    sum_time += (end - start);
  }

  T[idx] = sum_time;
  A[idx] = s[0] + s[1] + s[2] + s[3] + 
    s[4] + s[5] + s[6] + s[7];

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



__global__ void ILP_latency_fmad(int iteration, unsigned long long int * T, double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  register int start = 0;
  register int end = 0;
  unsigned long long sum_time = 0;

  register double a1 = 1;
  register double a2 = 2;
  register double a3 = 3;
  register double a4 = 4;
  register double a5 = 5;
  register double a6 = 6;
  register double a7 = 7;
  register double a8 = 8;
  register double a9 = 9;
  register double a10 = 10;
  for (int i = 0; i < iteration; i++) {
    asm volatile ("{\n\t"
                  "mov.u32 %0, %%clock;\n\t"
                  "fma.rn.f64 %1, %1, %1, %1;\n\t"
                  "fma.rn.f64 %2, %2, %2, %2;\n\t"
                  "fma.rn.f64 %3, %3, %3, %3;\n\t"
                  "fma.rn.f64 %4, %4, %4, %4;\n\t"
                  "fma.rn.f64 %5, %5, %5, %5;\n\t"
                  "fma.rn.f64 %6, %6, %6, %6;\n\t"
                  "fma.rn.f64 %7, %7, %7, %7;\n\t"
                  "fma.rn.f64 %8, %8, %8, %8;\n\t"
                  "fma.rn.f64 %9, %9, %9, %9;\n\t"
                  "fma.rn.f64 %10, %10, %10, %10;\n\t"
                  "mov.u32 %11, %%clock;\n\t"
                  "}"
                  :  "=r"(start), "=d"(a1),"=d"(a2),"=d"(a3), "=d"(a4),"=d"(a5),
		     "=d"(a6),"=d"(a7),"=d"(a8), "=d"(a9),"=d"(a10), "=r"(end) :: "memory"
                  );

    sum_time += (end - start);
  }
  T[idx] = sum_time;
  A[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10;

}




int main(){
  
  int SM = 15;
  int B = 1024;
  //int n = SM*B*32;
  int n = SM*B*16*100;
  double * A = new double[n];
  unsigned long long int * start = new unsigned long long int[n];
  unsigned long long int * end = new unsigned long long int[n];


  double * dA;
  unsigned long long int *dStart;
  unsigned long long int *dEnd;
  cudaMalloc(&dA, (n) * sizeof(double));
  cudaMalloc((void**)&dStart, n * sizeof(unsigned long long int));
  cudaMalloc((void**)&dEnd, n * sizeof(unsigned long long int));

  //register_latency<<<n/B, B>>>(1, dT);
  //ILP_latency_fmad<<<n/B, B>>>(10, dT, dA);
  


for (int i = 0; i < 1; i ++) {
    clock_t t = clock();
    gl2r_latency<<<SM, B, 49152>>>(50, dStart, dEnd, dA);
    cudaDeviceSynchronize();
    t = clock() - t;
    float real_time = ((float)t)/CLOCKS_PER_SEC;
    cout <<"Runing time: " << real_time << " s." << endl;
    long long total_byte = SM*B*64*29;
    double total_gb = total_byte/1e9;
    total_gb *= 50;
    cout << "Total data requested:"<<total_gb << " GB."<< endl; 
    double throughput = total_gb/real_time;
    cout <<"Throughput: " << throughput << " GB/s." << endl;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
  }

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
 cudaMemcpy(start, dStart, n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
 cudaMemcpy(end, dEnd, n * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
 
  
  
  if (false) {
    long long int base = LLONG_MAX;
    for (int i = 0; i < SM*B; i++) {
      base = min(base, start[i]);
    }
    cout << "base=" << base << endl;
    long long int sum = 0;
    for (int i = 0; i < SM*B; i++) {
      sum += end[i] - start[i];
      cout << "Thread.id[" << i << "]: latency "<< end[i] - start[i] << " cycles ("<< start[i]-base << " - " << end[i] - base << ")" << endl;
    }
    cout << "Average latency is: " << (double)sum / (SM*B) << "cycles" << endl;
  }
}

