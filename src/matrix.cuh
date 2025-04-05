#pragma once

#include <cuda_runtime.h>

cudaError_t compute_inverse_matrix(
	float* input,
	float* tmp,
	float* output,
	const int dimension
);
