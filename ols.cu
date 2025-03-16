#include "ols.cuh"

#include <stdio.h>
#include "device_launch_parameters.h"

#include "matrix.cuh"

#define CHECK_CUDA_ERROR(msg) { \
	if (m_cudaStatus != cudaSuccess) { \
		fprintf(stderr, msg, cudaGetErrorString(m_cudaStatus)); \
		return; \
	} \
}

#define CHECK_CUDA_NO_ERROR() { \
	if (m_cudaStatus != cudaSuccess) { \
		return; \
	} \
}

OLS::OLS(const int dimension) :
	m_dimension(dimension),
	m_cudaStatus(cudaSuccess),
	m_is_estimator_dirty(true),
	m_return(dimension)
{
	allocate_cuda_memory();
	CHECK_CUDA_NO_ERROR();

	clear_cuda_memory();
	CHECK_CUDA_NO_ERROR();
}

void OLS::allocate_cuda_memory()
{
	m_cudaStatus = cudaMalloc((void**)&m_sample, m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	m_cudaStatus = cudaMalloc((void**)&m_rhs, m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	m_cudaStatus = cudaMalloc((void**)&m_expected, 1 * sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	m_cudaStatus = cudaMalloc((void**)&m_mat_to_invert, m_dimension * m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	m_cudaStatus = cudaMalloc((void**)&m_inverted_mat, m_dimension * m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	m_cudaStatus = cudaMalloc((void**)&m_tmp_mat, m_dimension * m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");

	m_cudaStatus = cudaMalloc((void**)&m_output, m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMalloc failed! %s");
}

void OLS::clear_cuda_memory()
{
	m_is_estimator_dirty = true;

	m_cudaStatus = cudaMemset(m_sample, 0, m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMemset failed! %s");

	m_cudaStatus = cudaMemset(m_mat_to_invert, 0, m_dimension * m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMemset failed! %s");
}

void OLS::update_status_and_synchronize()
{
	m_cudaStatus = cudaGetLastError();
	CHECK_CUDA_ERROR("kernel launch failed: %s");

	m_cudaStatus = cudaDeviceSynchronize();
	CHECK_CUDA_ERROR("An error occured while executing cudaDeviceSynchronize: %s");
}

OLS::~OLS()
{
	cudaFree(m_output);
	cudaFree(m_sample);
	cudaFree(m_expected);
	cudaFree(m_tmp_mat);
	cudaFree(m_inverted_mat);
	cudaFree(m_mat_to_invert);
	cudaFree(m_rhs);
}

cudaError_t OLS::cudaStatus() const
{
	return m_cudaStatus;
}


/* Sample addition */
__global__ void add_sample_to_mat(
	const float* sample,
	float* mat_to_invert,
	int dimension
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < dimension && j < dimension)
		mat_to_invert[i * dimension + j] += sample[i] * sample[j];
}

__global__ void add_sample_to_rhs(
	const float* sample,
	const float* expected,
	float* rhs,
	int dimension
)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dimension)
		rhs[i] += *expected * sample[i];
}

void OLS::add_sample(const float* sample, const float expected)
{
	m_is_estimator_dirty = true;

	cudaMemcpy(m_sample, sample, m_dimension * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(m_expected, &expected, 1 * sizeof(float), cudaMemcpyHostToDevice);

	add_sample_to_mat <<<dim3((m_dimension + 31) / 32, (m_dimension + 31) / 32), dim3(32, 32)>>> (m_sample, m_mat_to_invert, m_dimension);
	update_status_and_synchronize();
	CHECK_CUDA_NO_ERROR();

	add_sample_to_rhs <<<dim3((m_dimension + 31) / 32, 1), dim3(32, 1)>>> (m_sample, m_expected, m_rhs, m_dimension);
	update_status_and_synchronize();
	CHECK_CUDA_NO_ERROR();
}

/* Estimator computation */
__global__ void retrieve_output(
	const float* rhs,
	const float* inverted_mat,
	float* output,
	int dimension,
	int i
)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (j >= dimension)
		return;

	output[i] += inverted_mat[i * dimension + j] * rhs[j];
}

void OLS::compute_output()
{
	// Compute the output
	m_cudaStatus = compute_inverse_matrix(
		m_mat_to_invert,
		m_tmp_mat,
		m_inverted_mat,
		m_dimension
	);
	CHECK_CUDA_NO_ERROR();

	m_cudaStatus = cudaMemset(m_output, 0, m_dimension * sizeof(float));
	CHECK_CUDA_ERROR("cudaMemset failed! %s");
	
	for (int i = 0; i < m_dimension; i++) {
		retrieve_output
			<<<dim3(1, (m_dimension + 31) / 32), dim3(1, 32)>>> (
				m_rhs, m_inverted_mat, m_output, m_dimension, i
			);
		update_status_and_synchronize();
		CHECK_CUDA_NO_ERROR();
	}

	cudaMemcpy(m_return.data(), m_output, m_dimension * sizeof(float), cudaMemcpyDeviceToHost);
}

const std::vector<float> OLS::retrieve_estimator()
{
	if (m_is_estimator_dirty) {
		compute_output();
		if (m_cudaStatus == cudaSuccess)
			m_is_estimator_dirty = false;
	}

	return m_return;
}
