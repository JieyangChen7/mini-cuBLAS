#include <iostream>
#include <stdio.h>
#include <climits>
#include <algorithm>
using namespace std;

__global__ void gl2rc_throughput(int iteration,double * A) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ double cache[];

  A = A + idx;
  register double * A1 = A;
  register double * A2 = A+1 * blockDim.x;
  register double * A3 = A+2 * blockDim.x;
  register double * A4 = A+3 * blockDim.x;
  register double * A5 = A+4 * blockDim.x;
  register double * A6 = A+5 * blockDim.x;
  register double * A7 = A+6 * blockDim.x;
  register double * A8 = A+7 * blockDim.x;
  register double * A9 = A+8 * blockDim.x;
  register double * A10 = A+9 * blockDim.x;
  register double * A11 = A+10 * blockDim.x;
  register double * A12 = A+11 * blockDim.x;
  register double * A13 = A+12 * blockDim.x;
  register double * A14 = A+13 * blockDim.x;
  register double * A15 = A+14 * blockDim.x;
  register double * A16 = A+15 * blockDim.x;
  register double * A17 = A+16 * blockDim.x;
  register double * A18 = A+17 * blockDim.x;
  register double * A19 = A+18 * blockDim.x;
  register double * A20 = A+19 * blockDim.x;
  register double * A21 = A+20 * blockDim.x;
  register double * A22 = A+21 * blockDim.x;
  register double * A23 = A+22 * blockDim.x;
  register double * A24 = A+23 * blockDim.x;
  register double * A25 = A+24 * blockDim.x;
  register double * A26 = A+25 * blockDim.x;
  register double * A27 = A+26 * blockDim.x;
  register double * A28 = A+27 * blockDim.x;
  register double * A29 = A+28 * blockDim.x;

  //register double * A30 = A+29 * blockDim.x;                                                               
  //register double * A31 = A+30 * blockDim.x;                                                                                              
  //register double * A32 = A+31 * blockDim.x;
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



int main(){

  int SM = 30;
  int B = 1024;
  int n = SM*B*32;                                                                                                                        
  
  double * A = new double[n];

  double * dA;
  cudaMalloc(&dA, (n) * sizeof(double));


  int iteration = 50;
  int access_per_iter = 29;
  clock_t t = clock();
  gl2rc_throughput<<<SM, B, 49152>>>(iteration, dA);
  cudaDeviceSynchronize();
  t = clock() - t;
  float real_time = ((float)t)/CLOCKS_PER_SEC;
  cout <<"Runing time: " << real_time << " s." << endl;
  long long total_byte = SM * B * sizeof(double) * access_per_iter;
  double total_gb = total_byte/1e9;
  total_gb *= iteration;
  cout << "Total data requested:"<<total_gb << " GB."<< endl;
  double throughput = total_gb/real_time;
  cout <<"Throughput: " << throughput << " GB/s." << endl;
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
      printf("Error: %s\n", cudaGetErrorString(err));


}
