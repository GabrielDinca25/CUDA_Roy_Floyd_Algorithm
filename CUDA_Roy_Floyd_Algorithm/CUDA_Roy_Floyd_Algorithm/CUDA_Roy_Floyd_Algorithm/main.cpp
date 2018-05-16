#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define SIZE 5;

__global__ void RoyFloyd(int* d_matrix, int k)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	
	if (d_matrix[i][j] > d_matrix[i][k] + d_matrix[k][j])
	{
		d_matrix[i][j] = d_matrix[i][k] + d_matrix[k][j];
	}
}

void main()
{
	int h_matrix[5][5] = { { 0,2,5,6,13 },{ 2, 0, 3, 4, 11 },{ 5, 3, 0, 1, 8 },{ 6, 4, 1, 0, 9 },{ 13,11,8,9,0 } };
	int* d_matrix;
	dim3 threadsPerBlock(SIZE, SIZE);

	cudaMalloc(&d_matrix, SIZE);
	cudaMemCpy(d_matrix, h_matrix, SIZE, cudaMemcpyHostToDevice);

	for (int k = 0; k < SIZE; k++)
	{
		RoyFloyd << < SIZE, threadsPerBlock >> > (d_matrix, k);
	}

	cudaMemCpy(h_matrix, d_matrix, SIZE, cudaMemCpyDeviceToHost);
	cudaFree(d_matrix); 

	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			cout << h_matrix[i][j] << " ";
		}
		cout << endl;
	}

	return 0;
}