#pragma once

#include <cuda_runtime.h>

cudaError_t compute_inverse_matrix(
	float* input,
	float* tmp,
	float* output,
	const int dimension
);

cudaError_t compute_cholesky_lower_matrix(
	float* input,
	float* output,
	const int N
);