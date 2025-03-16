#include "matrix.cuh"

#include <cstdio>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"

// CTI: Coordonates To Index
#define CTI(i, j) ((i) * N + (j))

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

__global__ void removeUpperHalf(
	float* mat,
	int N
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < j && i < N && j < N)
		mat[CTI(i, j)] = 0;
}

// We're using the Cholesky decomposition to compute the inverse of the matrix
// Because we're only dividing symmetric positive definite matrix
// The notation is based on this course: 
// https://courses.grainger.illinois.edu/cs554/fa2015/notes/07_cholesky.pdf
__global__ void cdiv(
	float* mat,
	float inv_sqrt_m_jj,
	int N,
	int j
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= j && i < N)
		mat[CTI(i, j)] *= inv_sqrt_m_jj;
}

__global__ void cmod(
	float* output,
	int N,
	int j,
	int k
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= j && i < N)
		output[CTI(i, j)] -= output[CTI(i, k)] * output[CTI(j, k)];
}

#define CHECK_CUDA_ERROR(msg) { \
	if (status != cudaSuccess) { \
		fprintf(stderr, "[" __FILE__ ":%d]" msg "\n", __LINE__, cudaGetErrorString(status)); \
		return status; \
	} \
}

cudaError_t compute_cholesky_lower_matrix(
	float* input,
	float* output,
	const int N
) {
	cudaError_t status;
	float inv_sqrt_m_kk;

	status = cudaMemcpy(output, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
	CHECK_CUDA_ERROR("cudaMemcpy failed! %s");

	// Compute the Cholesky lower decomposition
	for (int k = 0; k < N; k++) {
		status = cudaMemcpy(&inv_sqrt_m_kk, output + CTI(k, k), sizeof(float), cudaMemcpyDeviceToHost);
		CHECK_CUDA_ERROR("cudaMemcpy failed! %s");
		inv_sqrt_m_kk = rsqrtf(inv_sqrt_m_kk);

		cdiv << <dim3((N + 31) / 32, 1), dim3(32, 1) >> > (
			output, inv_sqrt_m_kk, N, k
			);
		status = cudaGetLastError();
		CHECK_CUDA_ERROR("kernel launch failed: %s");

		for (int j = k + 1; j < N; j++) {
			cmod << <dim3((N + 31) / 32, 1), dim3(32, 1) >> > (
				output, N, j, k
				);
			status = cudaGetLastError();
			CHECK_CUDA_ERROR("kernel launch failed: %s");
		}

		status = cudaDeviceSynchronize();
		CHECK_CUDA_ERROR("An error occured while executing cudaDeviceSynchronize: %s");
	}

	// Clean the upper part of the matrix
	removeUpperHalf << <dim3((N + 31) / 32, (N + 31) / 32), dim3(32, 32) >> > (
		output, N
	);

	status = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("An error occured while executing cudaDeviceSynchronize: %s");

	return status;
}


__global__ void backSubstitutionRowStep1(
	const float* input,
	float* output,
	int N,
	int k
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= N)
		return;

	if (j < k)
		output[CTI(k, i)] -= input[CTI(k, j)] * output[CTI(j, i)];
}

__global__ void backSubstitutionRowStep2(
	const float* input,
	float* output,
	int N,
	int k
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
		output[CTI(k, i)] /= input[CTI(k, k)];
}

__global__ void computeCholeskyInverseMatrixFromLowerInverse(
	const float* input,
	float* output,
	int N
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= N || j >= N)
		return;

	output[CTI(i, j)] = 0;
	for (int k = 0; k < N; k++)
		output[CTI(i, j)] += input[CTI(k, i)] * input[CTI(k, j)];
}

cudaError_t compute_inverse_matrix(
	float *input,
	float *tmp,
	float *output,
	const int N
) {
	cudaError_t status;

	status = compute_cholesky_lower_matrix(input, output, N);
	if (status != cudaSuccess)
		return status;

	// Compute the inverted matrix
	setToIdentity << <dim3((N + 31) / 32, (N + 31) / 32), dim3(32, 32) >> > (
		tmp, N
		);
	status = cudaGetLastError();
	CHECK_CUDA_ERROR("kernel launch failed: %s");

	for (int k = 0; k < N; k++) {
		// The blocks must be of size 1
		// Because of concurrency issues
		backSubstitutionRowStep1 << <dim3(N, N), dim3(1, 1) >> > (
			output, tmp, N, k
			);
		status = cudaGetLastError();
		CHECK_CUDA_ERROR("kernel launch failed: %s");
		
		backSubstitutionRowStep2 << <dim3((N + 31) / 32, 1), dim3(32, 1) >> > (
			output, tmp, N, k
			);
		status = cudaGetLastError();
		CHECK_CUDA_ERROR("kernel launch failed: %s");
	}

	computeCholeskyInverseMatrixFromLowerInverse << <dim3((N + 31) / 32, (N + 31) / 32), dim3(32, 32) >> > (
		tmp, output, N
		);

	return status;
}
