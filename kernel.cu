
#include "ols.cuh"
#include "matrix.cuh"
#include "gpu_timer.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <vector>

__declspec(noreturn) void die(const char *msg)
{
	fprintf(stderr, msg);
	exit(1);
}

void test_matrix_invert() {
	cudaError_t cudaStatus;
	const int N = 3;
	float* in_matrix, * tmp_matrix, * out_matrix;

	// values = X X^T where X = (1, 0.5, -1)
	float values[9] = {
		 4, -2,  0,
		-2,  2,  3,
		 0,  3, 10
	};

	cudaStatus = cudaMalloc((void**)&in_matrix, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess)
		die("Failed to allocate memory\n");

	cudaStatus = cudaMalloc((void**)&tmp_matrix, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess)
		die("Failed to allocate memory\n");

	cudaStatus = cudaMalloc((void**)&out_matrix, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess)
		die("Failed to allocate memory\n");

	cudaStatus = cudaMemcpy(in_matrix, values, N * N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		die("Failed to copy memory\n");

	cudaStatus = compute_inverse_matrix(in_matrix, tmp_matrix, out_matrix, N);
	if (cudaStatus != cudaSuccess)
		die("Failed to compute the inverse\n");

	/*
	Expected output:
	2.75	5	-1.5
	5		10		-3
	-1.5	-3		1
	*/
	cudaStatus = cudaMemcpy(values, out_matrix, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		die("Failed to copy memory\n");

	printf("Inverse matrix:\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
			printf("%f ", values[i * N + j]);
		printf("\n");
	}

	cudaFree(in_matrix);
	cudaFree(tmp_matrix);
	cudaFree(out_matrix);
}

int main(int argc, char* argv[])
{
	char* end;
	cudaError_t cudaStatus;

	if (argc != 2)
		die("Usage: ./kernel <dimension>\n");

	const int dimension = strtol(argv[1], &end, 10);
	if (end != argv[1] + strlen(argv[1]))
		die("Invalid argument\n");

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return -1;
	}

	// Test the matrix inversion
	//test_matrix_invert();

	// Test the Cuda OLS class
	{
		GpuTimer bench;
		std::vector<float> sample(dimension);
		float expected;

		OLS ols(dimension);
		if (ols.cudaStatus() != cudaSuccess)
			die("Failed to allocate memory\n");

		bench.Start();
		for (int i = 0; i < 1e4; i++) {
			// Build fake data
			for (int j = 0; j < dimension; j++)
				sample[j] = (float)rand() / RAND_MAX;
			expected = (float)rand() / RAND_MAX;

			ols.add_sample(sample.data(), expected);
			if (ols.cudaStatus() != cudaSuccess)
				return 1;
		}

		bench.PrintElapsed();

		std::vector<float> est = ols.retrieve_estimator();

		bench.PrintElapsed();
		/*
		for (float e : est)
			printf("%f ", e);
		*/
	}

	return 0;
}
