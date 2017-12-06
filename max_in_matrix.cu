#include <math.h> 
#include <iostream>
using namespace std;

__global__ void find_max_abs_diff_kernel(int m, int n, float * A, float * B, float * result)
{
	register float max_diff = 0.0;
	register int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
    	float * currA = A + idx;
    	float * currB = B + idx;

    	for (int i = 0; i < m; i++)
    	{
    		float a = *currA;
        	float b = *currB;
    		max_diff = fmaxf(max_diff, fabsf(a - b));
    		currA += n;
    		currB += n;
    	}
    }

    extern __shared__ float shared_array[];
    shared_array[threadIdx.x] = max_diff;
    for (int i = blockDim.x / 2; i >= 1; i /= 2)
    {
        if (threadIdx.x < i)
        {
            shared_array[threadIdx.x] = fmaxf(shared_array[threadIdx.x], shared_array[threadIdx.x + i]);
        }
        __syncthreads();
    }
    result[blockIdx.x] = shared_array[0];

}

float find_max_abs_diff(int m, int n, float * dA, float * dB) 
{
   
    int threadsPerBlock = 128; // must be power of 2
    int blocksPerGrid = ceil(n / threadsPerBlock);

    float * result = new float[blocksPerGrid];
    float * dResult;
    cudaMalloc(&dResult, blocksPerGrid * sizeof(float));

    find_max_abs_diff_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(m, n, dA, dB, dResult);
    cudaMemcpy(result, dResult, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float max = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        max = fmax(max, result[i]);
    }

    return max;

}


int main()
{
    int m = 1000;
    int n = 1000;
    float * A = new float[m * n];
    float * B = new float[m * n];

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            A[i * n + j] = 0;
            B[i * n + j] = i * n + j;
        }
    }


    float * dA;
    cudaMalloc(&dA, m * n * sizeof(float));

    float * dB; 
    cudaMalloc(&dB,  m * n * sizeof(float));

    cudaMemcpy(dA, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, m * n * sizeof(float), cudaMemcpyHostToDevice);

    float max = find_max_abs_diff(m, m, dA, dB);
    cout << max << endl;

}
