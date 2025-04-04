#include "matrix.cuh"

#include <cstdio>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

// CTI: Coordonates To Index
#define CTI(i, j) ((i) * N + (j))

#define CHECK_CUDA_ERROR(msg) { \
	if (status != cudaSuccess) { \
		fprintf(stderr, "[" __FILE__ ":%d]" msg "\n", __LINE__, cudaGetErrorString(status)); \
		return status; \
	} \
}

#define TRY_KERNEL(...) { \
	__VA_ARGS__; \
	status = cudaGetLastError(); \
	CHECK_CUDA_ERROR("Kernel launch failed: %s"); \
}

__global__ void setToIdentity(
	float* output,
	int N
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= N || j >= N)
		return;

	output[CTI(i, j)] = (i == j) ? 1 : 0;
}

__global__ void getMaxValue(
	float* mat,
	float* max_val,
	int* col_id,
	int N,
	int i
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float val = abs(mat[CTI(i, j)]);
	if(val > *max_val) {
		*max_val = val;
		*col_id = i;
	}
}

__global__ void swapRows(
	float* mat,
	int* col_id,
	int N,
	int k
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= N)
		return;

	float tmp = mat[CTI(k, j)];
	mat[CTI(k, j)] = mat[CTI(*col_id, j)];
	mat[CTI(*col_id, j)] = tmp;
}

__global__ void divideRow(
	float* mat,
	float* max_val,
	int N,
	int k
) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= N)
		return;

	mat[CTI(k, j)] /= *max_val;
}

cudaError_t compute_inverse_matrix(
	float *input,
	float *tmp,
	float *output,
	const int N
) {
	cudaError_t status;
	float *max_val;
	int *max_val_col;

	status = cudaMalloc((void**)&max_val, sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	status = cudaMalloc((void**)&max_val_col, sizeof(int));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	status = cudaMemcpy(tmp, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
	CHECK_CUDA_ERROR("cudaMemcpy failed! %s");

	// Compute the inverted matrix
	setToIdentity << <dim3((N + 31) / 32, (N + 31) / 32), dim3(32, 32) >> > (
		output, N
	);
	status = cudaGetLastError();
	CHECK_CUDA_ERROR("kernel launch failed: %s");

	for (int k = 0; k < N; k++) {
		status = cudaMemset(max_val, 0, sizeof(float));
		CHECK_CUDA_ERROR("cudaMemset failed! %s");

		TRY_KERNEL(
			getMaxValue << <dim3(N, 1), dim3(1, 1) >> > (
				tmp, max_val, max_val_col, N, k
			)
		);

		status = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("An error occured while executing cudaDeviceSynchronize: %s");
		
		TRY_KERNEL(
			swapRows << <dim3((N + 31) / 32, 1), dim3(32, 1) >> > (
				tmp, max_val_col, N, k
			)
		);
		TRY_KERNEL(
			divideRow << <dim3((N + 31) / 32, 1), dim3(32, 1) >> > (
				tmp, max_val, N, k
			)
		);
		
		TRY_KERNEL(
			swapRows << <dim3((N + 31) / 32, 1), dim3(32, 1) >> > (
				output, max_val_col, N, k
			)
		);
		TRY_KERNEL(
			divideRow << <dim3((N + 31) / 32, 1), dim3(32, 1) >> > (
				output, max_val, N, k
			)
		);

		
	}

	status = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("An error occured while executing cudaDeviceSynchronize: %s");

	status = cudaFree(max_val);
	CHECK_CUDA_ERROR("cudaFree failed! %s");

	status = cudaFree(max_val_col);
	CHECK_CUDA_ERROR("cudaFree failed! %s");

	return status;
}
